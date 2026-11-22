"""
3-Stage Training Script for LLaVA-Med + Layered Cross-Attention
(LEARNED-GATE VARIANT — control experiment)
================================================================

This is a copy of train_xattn.py with exactly one substitution: the model
class is LlavaMistralXAttnLearnedGateForCausalLM (learned tanh gate, same
zero-initialized W^O) instead of LlavaMistralXAttnForCausalLM (fixed
alpha=0.05). Everything else — dataset, freezing schedule, LR/warmup/
scheduler, batch size, grad accum, save strategy — is identical, so any
difference in the resulting loss curve / final metrics is attributable to
the fixed-alpha vs learned-gate choice alone.

Stage 1 – Cross-Attention Alignment
  Frozen : LLM (all layers) + Vision Tower
  Trained: cross_attn_modules only
  Goal   : align visual representations with text semantics

Usage:
  python train_xattn_learned_gate.py --stage 1 --base_model microsoft/llava-med-v1.5-mistral-7b
"""

import argparse
import os
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import json

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from llava.model.language_model.llava_mistral_xattn_learned_gate import (
    LlavaMistralXAttnLearnedGateForCausalLM,
    LlavaMistralXAttnLearnedGateConfig,
)
from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX, DEFAULT_IMAGE_TOKEN


# ---------------------------------------------------------------------------
# Dataset (identical to train_xattn.py)
# ---------------------------------------------------------------------------

class MedicalVQADataset(Dataset):
    """
    Expects a JSONL file where each line is:
      {
        "image": "relative/path/to/image.jpg",
        "question": "...",
        "answer": "..."
      }

    The image path is resolved relative to `image_root`.
    """

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
        print(f"[Dataset] {len(self.samples)} samples ready for training")

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
# Freezing helpers
# ---------------------------------------------------------------------------

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def set_stage1_trainable(model: LlavaMistralXAttnLearnedGateForCausalLM):
    """Stage 1: train cross-attention only."""
    freeze_all(model)
    for p in model.model.cross_attn_modules.parameters():
        p.requires_grad = True
    _print_trainable(model, "Stage 1")


def set_stage2_trainable(
    model: LlavaMistralXAttnLearnedGateForCausalLM,
    top_k_layers: int = 4,
):
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


def set_stage3_trainable(
    model: LlavaMistralXAttnLearnedGateForCausalLM,
    top_k_layers: int = 4,
):
    set_stage2_trainable(model, top_k_layers)
    _print_trainable(model, "Stage 3")


def _print_trainable(model, tag: str):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[{tag}] Trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.1f}%)"
    )


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(args):
    model_path = args.resume_from_checkpoint if args.resume_from_checkpoint else args.base_model
    is_resuming = bool(args.resume_from_checkpoint)

    print(f"Loading model from: {model_path}" + (" (resuming checkpoint)" if is_resuming else ""))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_resuming:
        config = LlavaMistralXAttnLearnedGateConfig.from_pretrained(model_path)
    else:
        config = LlavaMistralXAttnLearnedGateConfig.from_pretrained(
            model_path,
            cross_attn_num_heads=args.cross_attn_num_heads,
            cross_attn_dropout=args.cross_attn_dropout,
            cross_attn_sparse_topk=args.cross_attn_sparse_topk or None,
        )

    assert not (args.load_in_8bit and args.load_in_4bit), \
        "--load_in_8bit and --load_in_4bit are mutually exclusive"
    quantized = args.load_in_8bit or args.load_in_4bit

    if quantized:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        if args.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = LlavaMistralXAttnLearnedGateForCausalLM.from_pretrained(
            model_path,
            config=config,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        model = LlavaMistralXAttnLearnedGateForCausalLM.from_pretrained(
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
        model.model.build_cross_attention_modules(
            vision_dim=vision_dim, num_patches=num_patches
        )

    ca_train_dtype = torch.float32 if quantized else torch.bfloat16
    layer_index_map = model.model._layer_index_map()
    for layer_idx, module_pos in layer_index_map.items():
        layer_device = next(model.model.layers[layer_idx].parameters()).device
        model.model.cross_attn_modules[module_pos].to(device=layer_device, dtype=ca_train_dtype)

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
    p.add_argument("--base_model",   type=str,   required=True,
                   help="HF model ID or local path (e.g. microsoft/llava-med-v1.5-mistral-7b)")
    p.add_argument("--data_path",    type=str,   required=True,
                   help="Path to JSONL training file")
    p.add_argument("--image_root",   type=str,   required=True,
                   help="Root directory for images")
    p.add_argument("--output_dir",   type=str,   default="./output_learned_gate")
    p.add_argument("--top_k_layers", type=int,   default=4,
                   help="Number of top LLM layers to unfreeze in Stage 2/3")
    p.add_argument("--max_length",   type=int,   default=512)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--grad_accum",   type=int,   default=4)
    p.add_argument("--num_epochs",   type=int,   default=3)
    p.add_argument("--cross_attn_num_heads",   type=int,   default=8)
    p.add_argument("--cross_attn_dropout",     type=float, default=0.1)
    p.add_argument("--cross_attn_sparse_topk", type=int,   default=64,
                   help="Top-k visual tokens for sparse cross-attn in middle layers "
                        "(~11%% of 576 CLIP patches). Set to 0 for dense everywhere.")
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a saved checkpoint directory to resume training from.",
    )
    p.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load base LLM in 8-bit (bitsandbytes) to save ~7 GB VRAM.",
    )
    p.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load base LLM in 4-bit NF4 (bitsandbytes). Mutually exclusive with --load_in_8bit.",
    )
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for cross-attention module init, dropout, and "
                        "data shuffling. Use the SAME seed as the matching fixed-alpha "
                        "run to isolate the gate-vs-alpha variable.")
    p.add_argument("--save_steps", type=int, default=None,
                   help="If set, checkpoint every N optimizer steps instead of every epoch.")
    return p.parse_args()


LR_MAP = {
    1: 1e-5,
    2: 5e-5,
    3: 2e-5,
}

FREEZE_MAP = {
    1: set_stage1_trainable,
    2: set_stage2_trainable,
    3: set_stage3_trainable,
}


def main():
    args = parse_args()
    quantized = args.load_in_8bit or args.load_in_4bit

    set_seed(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    model, tokenizer, image_processor = load_model_and_tokenizer(args)

    freeze_fn = FREEZE_MAP[args.stage]
    if args.stage in (2, 3):
        freeze_fn(model, top_k_layers=args.top_k_layers)
    else:
        freeze_fn(model)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    dataset = MedicalVQADataset(
        data_path=args.data_path,
        image_root=args.image_root,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=args.max_length,
    )

    output_dir = os.path.join(args.output_dir, f"stage{args.stage}")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=LR_MAP[args.stage],
        warmup_steps=100,
        lr_scheduler_type="cosine",
        max_grad_norm=0.1,
        bf16=not quantized,
        fp16=quantized,
        optim="adamw_bnb_8bit",
        logging_steps=100,
        save_strategy="steps" if args.save_steps else "epoch",
        save_steps=args.save_steps if args.save_steps else 500,
        save_total_limit=1,
        save_only_model=True,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    _hidden_quant_flags = {}
    if quantized:
        for _flag in ("is_loaded_in_8bit", "is_loaded_in_4bit", "is_quantized"):
            if getattr(model, _flag, False):
                _hidden_quant_flags[_flag] = getattr(model, _flag)
                setattr(model, _flag, False)
        _hf_quantizer = getattr(model, "hf_quantizer", None)
        if _hf_quantizer is not None:
            _hidden_quant_flags["hf_quantizer"] = _hf_quantizer
            model.hf_quantizer = None

    class GateMonitor(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 500 == 0:
                gates = [torch.tanh(m.gate).item() for m in model.model.cross_attn_modules]
                print(f"  [XAttn-LearnedGate step={state.global_step}] "
                      f"gate(tanh)[0]={gates[0]:.4f}  "
                      f"mean={sum(gates)/len(gates):.4f}  "
                      f"max_abs={max(abs(g) for g in gates):.4f}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=make_collate_fn(tokenizer.pad_token_id),
        callbacks=[GateMonitor()],
    )

    for _flag, _value in _hidden_quant_flags.items():
        setattr(model, _flag, _value)

    print(f"\n=== Starting Stage {args.stage} training (LEARNED-GATE variant) ===")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    if quantized:
        import json as _json
        cfg_path = os.path.join(output_dir, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as _f:
                _cfg = _json.load(_f)
            if _cfg.pop("quantization_config", None) is not None:
                with open(cfg_path, "w") as _f:
                    _json.dump(_cfg, _f, indent=2)
                print(f"[Train] Stripped quantization_config from {cfg_path}")

    print(f"=== Stage {args.stage} complete (learned-gate variant). Saved to {output_dir} ===")


if __name__ == "__main__":
    main()
