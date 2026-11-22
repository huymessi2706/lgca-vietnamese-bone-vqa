"""
LoRA fine-tuning of the original LLaVA-Med model — ablation baseline, v3.

Why v3: v1 (naive 3-epoch) and v2 (anti-overfitting: lower rank, lower LR,
early stopping on val loss) both converged to EOS-degeneration — the model
predicts the end-of-sequence token as the very FIRST generated token on most
test samples, regardless of image/question. v2 even scored worse than v1
(BERTScore-F1 0.0327 vs 0.0514), which rules out "just needed better
hyperparameters" — but it does not yet rule out "the optimizer simply found
empty-string a free lunch because nothing in the loss specifically discourages
it". v3 closes that gap directly:

  - new --first_token_eos_penalty term: extra loss computed ONLY on the
    logits at the position predicting the first answer token. It penalizes
    the probability mass the model puts on <eos> at exactly that position.
    This does NOT touch EOS anywhere else in the sequence (the model is
    still free, and expected, to emit EOS at the real end of the answer —
    we are not breaking its ability to terminate, only its ability to
    terminate immediately).
  - everything else is inherited from v2: held-out val eval_loss, early
    stopping, low rank/LR, weight decay.

Interpreting the result either way is useful for the thesis:
  - if penalty fixes it    -> EOS-degeneration was the dominant failure mode,
                               worth documenting as its own finding.
  - if penalty still fails -> strongest possible evidence that the failure is
                               architectural (token-concat + this base model),
                               not a fixable training artifact.

Usage:
  python train_lora_baseline_v3.py \
      --base_model microsoft/llava-med-v1.5-mistral-7b \
      --data_path  bone_data/dataset_train_minimal.jsonl \
      --val_data_path bone_data/dataset_val_minimal.jsonl \
      --image_root ./bone_data \
      --output_dir ./outputs_lora_baseline_v3 \
      --num_epochs 6 \
      --lora_r 8 \
      --first_token_eos_penalty 1.0
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoTokenizer, EarlyStoppingCallback, Trainer,
                           TrainerCallback, TrainingArguments, set_seed)


class TimeLimitCallback(TrainerCallback):
    """Stops training once wall-clock training time exceeds max_seconds.

    Added for the equal-training-time-budget comparison (LoRA r=256 vs
    LGCA under the same wall-clock budget rather than the same step/epoch
    count). Inert when max_seconds is None -- does not change any existing
    run's behavior.
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

from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token


# ---------------------------------------------------------------------------
# Dataset (identical to v1/v2)
# ---------------------------------------------------------------------------

class LoRAVQADataset(Dataset):
    def __init__(self, data_path, image_root, tokenizer, image_processor, max_length=512):
        from llava.conversation import conv_templates

        with open(data_path, encoding="utf-8") as f:
            all_samples = [json.loads(l) for l in f if l.strip()]

        self.image_root      = image_root
        self.tokenizer       = tokenizer
        self.image_processor = image_processor
        self.max_length      = max_length

        valid = []
        skipped_img = 0
        skipped_trunc = 0
        for s in all_samples:
            if not os.path.exists(os.path.join(image_root, s["image"])):
                skipped_img += 1
                continue
            conv_p = conv_templates["mistral_instruct"].copy()
            conv_p.append_message(conv_p.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + s["question"])
            conv_p.append_message(conv_p.roles[1], None)
            prompt_ids = tokenizer_image_token(conv_p.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX)
            if len(prompt_ids) >= max_length - 1:
                skipped_trunc += 1
                continue
            valid.append(s)

        self.samples = valid
        if skipped_img:
            print(f"[Dataset] Skipped {skipped_img} samples with missing images")
        if skipped_trunc:
            print(f"[Dataset] Skipped {skipped_trunc} samples where question fills max_length")
        print(f"[Dataset] {len(self.samples)} samples ready ({data_path})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        question = sample["question"]
        answer   = sample["answer"]

        from llava.conversation import conv_templates

        conv_f = conv_templates["mistral_instruct"].copy()
        conv_f.append_message(conv_f.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv_f.append_message(conv_f.roles[1], answer)
        full = conv_f.get_prompt()

        conv_p = conv_templates["mistral_instruct"].copy()
        conv_p.append_message(conv_p.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv_p.append_message(conv_p.roles[1], None)
        prompt = conv_p.get_prompt()

        full_ids   = tokenizer_image_token(full,   self.tokenizer, IMAGE_TOKEN_INDEX)
        prompt_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX)

        full_ids   = full_ids[:self.max_length]
        input_ids  = torch.tensor(full_ids, dtype=torch.long)
        labels     = input_ids.clone()
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels[:prompt_len] = IGNORE_INDEX

        img_path     = os.path.join(self.image_root, sample["image"])
        image        = Image.open(img_path).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        return {
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": torch.ones_like(input_ids),
            "images":         pixel_values,
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

def make_collate_fn(pad_token_id: int):
    def collate_fn(batch):
        input_ids      = pad_sequence([b["input_ids"]      for b in batch], batch_first=True, padding_value=pad_token_id)
        labels         = pad_sequence([b["labels"]         for b in batch], batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
        images         = torch.stack([b["images"]          for b in batch])
        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": attention_mask, "images": images}
    return collate_fn


# ---------------------------------------------------------------------------
# Trainer with first-token anti-EOS penalty
# ---------------------------------------------------------------------------

class AntiEOSTrainer(Trainer):
    """
    Trainer that adds a penalty discouraging <eos> probability specifically at
    the position predicting the FIRST answer token (i.e. right after the
    prompt). Does not touch EOS anywhere else — the model can and should
    still emit EOS at the genuine end of an answer.

    For causal LM, the logits at position (i) predict the token at label
    position (i+1) (standard shift used internally by HF loss). So for each
    sample, we take:
        first_label_pos = index of first label != IGNORE_INDEX
        logit_pos        = first_label_pos - 1
    and penalize softmax(logits[logit_pos])[eos_token_id].

    Also skips/clamps NaN-or-Inf loss steps (same safety net as v1/v2).
    """

    def __init__(self, *args, eos_token_id: int, first_token_eos_penalty: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.eos_token_id = eos_token_id
        self.first_token_eos_penalty = first_token_eos_penalty

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        if self.first_token_eos_penalty > 0:
            labels = inputs["labels"]                 # (B, L)
            logits = outputs.logits                    # (B, L, V)
            mask = labels != IGNORE_INDEX
            has_label = mask.any(dim=1)
            if has_label.any():
                # argmax on a 0/1 float mask returns the index of the FIRST
                # occurrence of the max value, i.e. the first True.
                first_label_pos = mask.float().argmax(dim=1)
                logit_pos = (first_label_pos - 1).clamp(min=0)
                batch_idx = torch.arange(logits.size(0), device=logits.device)
                first_logits = logits[batch_idx, logit_pos, :].float()
                probs = F.softmax(first_logits, dim=-1)
                p_eos = probs[:, self.eos_token_id]
                p_eos = p_eos[has_label]
                penalty = -torch.log(1.0 - p_eos + 1e-6).mean()
                loss = loss + self.first_token_eos_penalty * penalty

        if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
            loss = torch.tensor(0.0, requires_grad=True, device=outputs.logits.device)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",     required=True)
    p.add_argument("--data_path",      required=True)
    p.add_argument("--val_data_path",  required=True)
    p.add_argument("--image_root",     required=True)
    p.add_argument("--output_dir",     default="./outputs_lora_baseline_v3")
    p.add_argument("--num_epochs",     type=int,   default=6)
    p.add_argument("--batch_size",     type=int,   default=1)
    p.add_argument("--grad_accum",     type=int,   default=16)
    p.add_argument("--eval_batch_size", type=int,  default=1)
    p.add_argument("--max_length",     type=int,   default=512)
    p.add_argument("--lora_r",         type=int,   default=8)
    p.add_argument("--lora_alpha",     type=int,   default=16)
    p.add_argument("--lora_dropout",   type=float, default=0.1)
    p.add_argument("--learning_rate",  type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=0.01)
    p.add_argument("--early_stopping_patience", type=int, default=2)
    p.add_argument("--first_token_eos_penalty", type=float, default=1.0,
                   help="Weight of the anti-EOS-at-first-token loss term. 0 disables it "
                        "(falls back to plain v2 behaviour).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for LoRA adapter init, dropout, and data shuffling. "
                        "Default 42 reproduces the original reported run.")
    p.add_argument("--load_in_8bit", action="store_true",
                   help="Load base LLM in 8-bit (bitsandbytes). The original AWS run used "
                        "full bf16 on a 24GB A10G; a 16GB GPU (e.g. Kaggle T4) needs this "
                        "to fit the 7B model plus LoRA plus activations.")
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Load base LLM in 4-bit NF4 (QLoRA-style, bitsandbytes). More memory-"
                        "efficient than --load_in_8bit's LLM.int8() outlier decomposition, "
                        "which can still OOM a 16GB GPU for a 7B model. Mutually exclusive "
                        "with --load_in_8bit.")
    p.add_argument("--save_steps", type=int, default=None,
                   help="If set, checkpoint every N optimizer steps (and eval at the same "
                        "cadence, required by load_best_model_at_end) instead of once per "
                        "epoch. Needed on time-limited sessions (e.g. Kaggle) where a single "
                        "epoch may not finish before the session is cut off.")
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to a checkpoint-N directory to resume training from.")
    p.add_argument("--max_train_seconds", type=float, default=None,
                   help="If set, stop training once wall-clock training time exceeds "
                        "this many seconds, regardless of --num_epochs. Used for the "
                        "equal-training-time-budget comparison against LGCA. None "
                        "(default) preserves the original fixed-epoch behavior.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Must run before model load: the LoRA adapter matrices are freshly
    # initialized (not from the LLaVA-Med checkpoint), so the seed here
    # controls their init in addition to dropout and data shuffling.
    set_seed(args.seed)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM

    print(f"Loading base model: {args.base_model}")
    if args.load_in_8bit or args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        if args.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = LlavaMistralForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quant_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        model = LlavaMistralForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device="cuda:0", dtype=torch.bfloat16)
    image_processor = vision_tower.image_processor

    # Free disk space: the base model's downloaded HF cache (~15GB fp16 weights)
    # is no longer needed once everything above is loaded into GPU memory, and
    # on quota-limited disks (e.g. Kaggle's /kaggle/working) it can be the
    # difference between a checkpoint save succeeding or failing with
    # "No space left on device" mid-training.
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

    if not (args.load_in_8bit or args.load_in_4bit):
        for param in model.parameters():
            param.requires_grad = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    train_dataset = LoRAVQADataset(
        data_path=args.data_path,
        image_root=args.image_root,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=args.max_length,
    )
    eval_dataset = LoRAVQADataset(
        data_path=args.val_data_path,
        image_root=args.image_root,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=args.max_length,
    )

    import inspect
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_strategy_kwarg = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=not args.load_in_8bit,
        fp16=args.load_in_8bit,
        optim="adamw_bnb_8bit",
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "epoch",
        save_steps=args.save_steps if args.save_steps else 500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        **{eval_strategy_kwarg: "steps" if args.save_steps else "epoch",
           **({"eval_steps": args.save_steps} if args.save_steps else {})},
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    if args.max_train_seconds is not None:
        callbacks.append(TimeLimitCallback(args.max_train_seconds))

    trainer = AntiEOSTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=make_collate_fn(tokenizer.pad_token_id),
        callbacks=callbacks,
        eos_token_id=tokenizer.eos_token_id,
        first_token_eos_penalty=args.first_token_eos_penalty,
    )

    print(f"\n=== LoRA baseline v3 training (up to {args.num_epochs} epochs, "
          f"r={args.lora_r}, lr={args.learning_rate}, "
          f"first_token_eos_penalty={args.first_token_eos_penalty}) ===")
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print(f"\nBest checkpoint: {trainer.state.best_model_checkpoint} "
          f"(eval_loss={trainer.state.best_metric:.4f})")

    if args.load_in_8bit or args.load_in_4bit:
        # PEFT cannot merge LoRA deltas into bitsandbytes 8-bit/4-bit quantized
        # layers (no dequantize-merge-requantize path here) — save the adapter
        # only. Downstream evaluation must load the base model + this adapter
        # directly (e.g. via peft.PeftModel.from_pretrained) instead of a
        # single merged dir.
        print("\n--load_in_8bit/--load_in_4bit set: skipping merge_and_unload "
              "(unsupported for quantized base layers). Saving LoRA adapter only.")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"=== Done. LoRA adapter (best-on-val) saved → {args.output_dir} ===")
    else:
        print("\nMerging LoRA weights into base model ...")
        merged = model.merge_and_unload()
        merged.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"=== Done. Merged model (best-on-val) saved → {args.output_dir} ===")


if __name__ == "__main__":
    main()
