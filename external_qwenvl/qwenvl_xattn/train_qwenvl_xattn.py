"""
Train Layered Gated Cross-Attention on top of a frozen Qwen2-VL-7B-Instruct.

Purpose (see modeling_qwen2vl_xattn.py docstring for full rationale): isolate
whether cross-attention beats token concatenation *independent of base model
quality*. Qwen2-VL already succeeds with concatenation (zero-shot BERTScore
0.86, LoRA-FT 0.91 on this Vietnamese dataset) — if XAttn improves on that,
it's strong evidence the fusion mechanism is generically better, not just a
rescue mechanism for a language-incapable base model (as LLaVA-Med was).

Run in the SAME "qwenvl" conda env used for train_lora_qwenvl.py (NOT "llava" —
that env's transformers==4.36.2 predates Qwen2-VL support entirely).

KNOWN RISK: untested against your exact server transformers/qwen-vl-utils
version — see modeling_qwen2vl_xattn.py's "KNOWN RISK" section. Run
--smoke_test first (1 sample, 1 forward+backward step, no Trainer) before a
full training job.

Usage:
  # 1. Smoke test first (always do this before a real run)
  python train_qwenvl_xattn.py --smoke_test \
      --base_model Qwen/Qwen2-VL-7B-Instruct \
      --data_path  bone_data/dataset_train_minimal.jsonl \
      --image_root ./bone_data

  # 2. Full training (Stage-1-only recipe: freeze everything except cross-attn)
  python train_qwenvl_xattn.py \
      --base_model Qwen/Qwen2-VL-7B-Instruct \
      --data_path  bone_data/dataset_train_minimal.jsonl \
      --val_data_path bone_data/dataset_val_minimal.jsonl \
      --image_root ./bone_data \
      --output_dir ./outputs_qwenvl_xattn_minimal \
      --num_epochs 3
"""

import argparse
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoProcessor, EarlyStoppingCallback, Trainer, TrainingArguments

from dataset import QwenVLXAttnDataset, make_collate_fn
from modeling_qwen2vl_xattn import Qwen2VLXAttnForCausalLM


class NaNSafeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
            loss = torch.tensor(0.0, requires_grad=True, device=loss.device)
        return (loss, outputs) if return_outputs else loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2-VL-7B-Instruct")
    p.add_argument("--data_path", required=True)
    p.add_argument("--val_data_path", default=None,
                   help="If given, evaluates each epoch and keeps the best checkpoint. "
                        "Omit for a quick first run (matches v1's naive recipe).")
    p.add_argument("--image_root", required=True)
    p.add_argument("--output_dir", default="./outputs_qwenvl_xattn_minimal")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--early_stopping_patience", type=int, default=2)
    p.add_argument("--smoke_test", action="store_true",
                   help="Run a single forward+backward pass on 1 sample, no Trainer, "
                        "then exit. Use this first to catch AttributeError/shape "
                        "mismatches cheaply before a multi-hour job.")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"Loading base model: {args.base_model}")
    model = Qwen2VLXAttnForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
    )
    processor = AutoProcessor.from_pretrained(
        args.base_model, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Freeze everything except cross-attn modules (Stage-1-only recipe,
    # same as llava/model/language_model/llava_mistral_xattn.py) ----
    for param in model.parameters():
        param.requires_grad = False
    for param in model.cross_attn_modules.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Params] trainable={n_trainable:,} / total={n_total:,} "
          f"({100 * n_trainable / n_total:.3f}%)")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    train_dataset = QwenVLXAttnDataset(
        data_path=args.data_path, image_root=args.image_root,
        processor=processor, max_length=args.max_length,
    )

    if args.smoke_test:
        print("\n=== SMOKE TEST: 1 sample, forward + backward, no Trainer ===")
        sample = train_dataset[0]
        batch = make_collate_fn(tokenizer.pad_token_id)([sample])
        batch = {k: v.to("cuda:0") for k, v in batch.items()}
        model.train()
        out = model(**batch)
        print(f"loss = {out.loss.item():.4f}")
        out.loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.cross_attn_modules.parameters()
                      if p.grad is not None]
        print(f"cross_attn grad norms: min={min(grad_norms):.2e} max={max(grad_norms):.2e} "
              f"(n_params_with_grad={len(grad_norms)})")
        print("=== SMOKE TEST PASSED ===")
        return

    eval_dataset = None
    if args.val_data_path:
        eval_dataset = QwenVLXAttnDataset(
            data_path=args.val_data_path, image_root=args.image_root,
            processor=processor, max_length=args.max_length,
        )

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_strategy_kwarg = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
    extra_eval_kwargs = {}
    if eval_dataset is not None:
        extra_eval_kwargs = {
            eval_strategy_kwarg: "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=True,
        optim="adamw_bnb_8bit",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
        **extra_eval_kwargs,
    )

    callbacks = []
    if eval_dataset is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = NaNSafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=make_collate_fn(tokenizer.pad_token_id),
        callbacks=callbacks,
    )

    print(f"\n=== Qwen2-VL + XAttn training ({args.num_epochs} epochs, "
          f"lr={args.learning_rate}, eval={'on' if eval_dataset else 'off'}) ===")
    trainer.train()

    print(f"\nSaving model -> {args.output_dir}")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("=== Done. ===")


if __name__ == "__main__":
    main()
