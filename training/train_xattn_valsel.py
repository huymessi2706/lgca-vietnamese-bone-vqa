"""
Same as train_xattn.py, with one addition: validation-based best-checkpoint
selection, matching train_lora_baseline_v3.py's protocol exactly, so the
parameter-matched comparison (LoRA r=256 vs. LGCA) no longer has an
asymmetric checkpoint-selection rule between the two configurations.
New file, does not modify train_xattn.py / train_xattn_root.py.

Added:
  --val_data_path       (required) separate held-out validation JSONL,
                        evaluated every --save_steps steps; best-on-val
                        checkpoint (lowest eval_loss) is the one saved and
                        reported, via load_best_model_at_end=True.
  --early_stopping_patience  same meaning and default (20 evaluations) as
                        train_lora_baseline_v3.py. With save_steps=250 over
                        1275 total steps there are only ~5 evaluations, so
                        patience=20 never actually triggers an early stop;
                        the only real effect is best-of-~5 checkpoint
                        selection, identical in spirit to the LoRA side.

Everything else (model, freezing, zone schedule, hyperparameters, data
loading, disk-cache clearing) is unchanged from train_xattn.py.
"""

import argparse
import os
import time
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import json

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    TrainingArguments,
    set_seed,
)

from llava.model import LlavaMistralXAttnForCausalLM, LlavaMistralXAttnConfig
from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX, DEFAULT_IMAGE_TOKEN


# ---------------------------------------------------------------------------
# Dataset (identical to train_xattn.py)
# ---------------------------------------------------------------------------

class MedicalVQADataset(Dataset):
    def __init__(
        self,
        data_path: str,
        image_root: str,
        tokenizer,
        image_processor,
        max_length: int = 512,
    ):
        with open(data_path) as f:
            all_samples = [json.loads(l) for l in f]
        self.image_root     = image_root
        self.tokenizer      = tokenizer
        self.image_processor = image_processor
        self.max_length     = max_length

        have_image = [
            s for s in all_samples
            if os.path.exists(os.path.join(image_root, s["image"]))
        ]
        skipped_img = len(all_samples) - len(have_image)
        if skipped_img:
            print(f"[Dataset] Skipped {skipped_img} samples with missing images")

        self.samples = []
        skipped_trunc = 0
        for s in have_image:
            prompt = f"{DEFAULT_IMAGE_TOKEN}\n{s['question']}\nAnswer:"
            plen = len(tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"])
            if plen < max_length:
                self.samples.append(s)
            else:
                skipped_trunc += 1
        if skipped_trunc:
            print(f"[Dataset] Skipped {skipped_trunc} samples where question fills max_length "
                  f"(no answer tokens after truncation)")
        print(f"[Dataset] {len(self.samples)} samples ready")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_path = os.path.join(self.image_root, sample["image"])
        image    = Image.open(img_path).convert("RGB")
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        question = sample["question"]
        answer   = sample["answer"]

        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{question}\nAnswer:"
        full   = f"{DEFAULT_IMAGE_TOKEN}\n{question}\nAnswer: {answer}{self.tokenizer.eos_token}"

        full_enc   = self.tokenizer(
            full, truncation=True, max_length=self.max_length, return_tensors="pt",
        )
        prompt_enc = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length, return_tensors="pt",
        )

        input_ids = full_enc.input_ids.squeeze(0)
        labels    = input_ids.clone()

        prompt_len        = prompt_enc.input_ids.shape[1]
        labels[:prompt_len] = IGNORE_INDEX

        return {
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": full_enc.attention_mask.squeeze(0),
            "images":         pixel_values,
        }


def make_collate_fn(pad_token_id: int):
    from torch.nn.utils.rnn import pad_sequence

    def collate_fn(batch):
        input_ids      = pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_token_id
        )
        labels         = pad_sequence(
            [b["labels"] for b in batch], batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = pad_sequence(
            [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
        )
        images = torch.stack([b["images"] for b in batch])

        return {
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": attention_mask,
            "images":         images,
        }

    return collate_fn


# ---------------------------------------------------------------------------
# Freezing helpers (identical to train_xattn.py)
# ---------------------------------------------------------------------------

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def set_stage1_trainable(model):
    freeze_all(model)
    for p in model.model.cross_attn_modules.parameters():
        p.requires_grad = True
    _print_trainable(model, "Stage 1")


def set_stage2_trainable(model, top_k_layers: int = 4):
    freeze_all(model)
    for p in model.model.cross_attn_modules.parameters():
        p.requires_grad = True
    num_layers = len(model.model.layers)
    for i in range(num_layers - top_k_layers, num_layers):
        for p in model.model.layers[i].parameters():
            p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True
    _print_trainable(model, f"Stage 2 (top {top_k_layers} layers)")


def set_stage3_trainable(model, top_k_layers: int = 4):
    set_stage2_trainable(model, top_k_layers)
    _print_trainable(model, "Stage 3")


def _print_trainable(model, tag: str):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{tag}] Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")


# ---------------------------------------------------------------------------
# Model loader (identical to train_xattn.py)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(args):
    model_path = args.resume_from_checkpoint if args.resume_from_checkpoint else args.base_model
    is_resuming = bool(args.resume_from_checkpoint)

    print(f"Loading model from: {model_path}" + (" (resuming checkpoint)" if is_resuming else ""))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_resuming:
        config = LlavaMistralXAttnConfig.from_pretrained(model_path)
    else:
        config = LlavaMistralXAttnConfig.from_pretrained(
            model_path,
            cross_attn_num_heads=args.cross_attn_num_heads,
            cross_attn_dropout=args.cross_attn_dropout,
            cross_attn_sparse_topk=args.cross_attn_sparse_topk or None,
        )

    model = LlavaMistralXAttnForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(dtype=torch.bfloat16)
    image_processor = vision_tower.image_processor

    if len(model.model.cross_attn_modules) > 0:
        model.model.rebuild_hooks()
    else:
        vision_dim  = vision_tower.hidden_size
        num_patches = vision_tower.num_patches
        model.model.build_cross_attention_modules(vision_dim=vision_dim, num_patches=num_patches)

    layer_index_map = model.model._layer_index_map()
    for layer_idx, module_pos in layer_index_map.items():
        layer_device = next(model.model.layers[layer_idx].parameters()).device
        model.model.cross_attn_modules[module_pos].to(device=layer_device, dtype=torch.bfloat16)

    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        commit_hashes = [
            revision.commit_hash
            for repo in cache_info.repos if repo.repo_id == args.base_model
            for revision in repo.revisions
        ]
        if commit_hashes:
            delete_strategy = cache_info.delete_revisions(*commit_hashes)
            print(f"[Disk] Freeing {delete_strategy.expected_freed_size / 1e9:.2f} GB "
                  f"by clearing HF cache for {args.base_model}")
            delete_strategy.execute()
    except Exception as e:
        print(f"[Disk] Warning: failed to clear HF cache for {args.base_model}: {e}")

    return model, tokenizer, image_processor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage",        type=int,   required=True,  choices=[1, 2, 3])
    p.add_argument("--base_model",   type=str,   required=True)
    p.add_argument("--data_path",    type=str,   required=True)
    p.add_argument("--val_data_path", type=str,  required=True,
                   help="Held-out validation JSONL, same schema as --data_path. "
                        "Evaluated every --save_steps steps; the lowest-eval_loss "
                        "checkpoint is kept (load_best_model_at_end), matching "
                        "train_lora_baseline_v3.py's protocol for a symmetric "
                        "parameter-matched comparison.")
    p.add_argument("--image_root",   type=str,   required=True)
    p.add_argument("--output_dir",   type=str,   default="./output")
    p.add_argument("--top_k_layers", type=int,   default=4)
    p.add_argument("--max_length",   type=int,   default=512)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--grad_accum",   type=int,   default=4)
    p.add_argument("--num_epochs",   type=int,   default=3)
    p.add_argument("--cross_attn_num_heads",   type=int,   default=8)
    p.add_argument("--cross_attn_dropout",     type=float, default=0.1)
    p.add_argument("--cross_attn_sparse_topk", type=int,   default=64)
    p.add_argument("--resume_from_checkpoint", type=str,   default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_steps", type=int, default=250,
                   help="Also used as the evaluation cadence (eval_steps=save_steps), "
                        "matching train_lora_baseline_v3.py's default usage in this project.")
    p.add_argument("--early_stopping_patience", type=int, default=20,
                   help="Same default as train_lora_baseline_v3.py. With save_steps=250 "
                        "over a short schedule this typically never triggers; the real "
                        "effect is best-of-N checkpoint selection via load_best_model_at_end.")
    p.add_argument("--max_train_seconds", type=float, default=None,
                   help="If set, stop training once wall-clock training time exceeds "
                        "this many seconds, regardless of --num_epochs. Used for the "
                        "equal-training-time-budget comparison against LoRA (r=256). "
                        "None (default) preserves the original fixed-epoch behavior.")
    return p.parse_args()


LR_MAP = {1: 1e-5, 2: 5e-5, 3: 2e-5}
FREEZE_MAP = {1: set_stage1_trainable, 2: set_stage2_trainable, 3: set_stage3_trainable}


def main():
    args = parse_args()

    set_seed(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    model, tokenizer, image_processor = load_model_and_tokenizer(args)

    freeze_fn = FREEZE_MAP[args.stage]
    if args.stage in (2, 3):
        freeze_fn(model, top_k_layers=args.top_k_layers)
    else:
        freeze_fn(model)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    train_dataset = MedicalVQADataset(
        data_path=args.data_path,
        image_root=args.image_root,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=args.max_length,
    )
    val_dataset = MedicalVQADataset(
        data_path=args.val_data_path,
        image_root=args.image_root,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=args.max_length,
    )

    output_dir = os.path.join(args.output_dir, f"stage{args.stage}")

    ta_params = TrainingArguments.__init__.__code__.co_varnames
    eval_strategy_kwarg = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=LR_MAP[args.stage],
        warmup_steps=100,
        lr_scheduler_type="cosine",
        max_grad_norm=0.1,
        bf16=True,
        optim="adamw_bnb_8bit",
        logging_steps=100,
        **{eval_strategy_kwarg: "steps"},
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    class GateMonitor(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 500 == 0:
                alphas = [m.alpha for m in model.model.cross_attn_modules]
                print(f"  [XAttn step={state.global_step}] alpha={alphas[0]:.4f} (fixed)")

    class TimeLimitCallback(TrainerCallback):
        """Stops training once wall-clock training time exceeds max_seconds.

        Mirrors train_lora_baseline_v3.py's TimeLimitCallback, for the
        equal-training-time-budget comparison (LGCA given the SAME wall-clock
        time LoRA r=256 took, to see whether extra training time closes the
        gap or LGCA plateaus -- isolating a training-time artifact from an
        architectural capacity limit). Inert when max_seconds is None.
        """

        def __init__(self, max_seconds: float):
            self.max_seconds = max_seconds
            self._start = None

        def on_train_begin(self, args, state, control, **kwargs):
            self._start = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            if self._start is not None and time.monotonic() - self._start >= self.max_seconds:
                control.should_training_stop = True
            return control

    callbacks = [GateMonitor(), EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    if args.max_train_seconds is not None:
        callbacks.append(TimeLimitCallback(args.max_train_seconds))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=make_collate_fn(tokenizer.pad_token_id),
        callbacks=callbacks,
    )

    print(f"\n=== Starting Stage {args.stage} training (validation-selected checkpoint) ===")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"=== Stage {args.stage} complete. Best-on-val checkpoint saved to {output_dir} ===")


if __name__ == "__main__":
    main()
