"""
3-Stage Training Script for LLaVA-Med + Layered Cross-Attention
================================================================

Stage 1 – Cross-Attention Alignment
  Frozen : LLM (all layers) + Vision Tower
  Trained: cross_attn_modules only
  Goal   : align visual representations with text semantics

Stage 2 – Partial Fine-Tuning
  Frozen : Vision Tower
  Trained: cross_attn_modules + top K LLM layers (+ optional LoRA on LLM)
  Goal   : task-aware vision-language reasoning

Stage 3 – Instruction Tuning (optional)
  Format : instruction-following  <image> Instruction: ... Answer: ...
  Trained: same as Stage 2, lower LR

Usage:
  python train_xattn.py --stage 1 --base_model microsoft/llava-med-v1.5-mistral-7b
  python train_xattn.py --stage 2 --base_model ./output/stage1 --top_k_layers 4
  python train_xattn.py --stage 3 --base_model ./output/stage2
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    CLIPImageProcessor,
    set_seed,
)

from llava.model import LlavaMistralXAttnForCausalLM, LlavaMistralXAttnConfig
from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX, DEFAULT_IMAGE_TOKEN


# ---------------------------------------------------------------------------
# Dataset
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

        # Drop samples whose image file is missing on disk
        have_image = [
            s for s in all_samples
            if os.path.exists(os.path.join(image_root, s["image"]))
        ]
        skipped_img = len(all_samples) - len(have_image)
        if skipped_img:
            print(f"[Dataset] Skipped {skipped_img} samples with missing images")

        # Drop samples whose question is so long that the answer is fully truncated.
        # When prompt_len >= max_length, all labels are IGNORE_INDEX → loss = NaN →
        # NaN gradients silently corrupt model weights even though loss.backward() runs.
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

        # ---- image ----
        img_path = os.path.join(self.image_root, sample["image"])
        image    = Image.open(img_path).convert("RGB")
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)   # (C, H, W)

        # ---- text ----
        question = sample["question"]
        answer   = sample["answer"]

        # DEFAULT_IMAGE_TOKEN acts as the visual anchor in the sequence.
        # LLaVA-Med was pretrained with <image> in the prompt; omitting it
        # breaks the alignment between the text embedding stream and the
        # cross-attention visual context.
        #
        # The `question` field already contains full clinical context from
        # preprocess_data.py (chief_complaint, history, bone_exam, lab findings,
        # xray_findings) plus the actual question — so no "Question:" wrapper needed.
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{question}\nAnswer:"
        full   = f"{DEFAULT_IMAGE_TOKEN}\n{question}\nAnswer: {answer}{self.tokenizer.eos_token}"

        # Tokenize full sequence and prompt separately to build label mask
        full_enc   = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = full_enc.input_ids.squeeze(0)
        labels    = input_ids.clone()

        # Mask prompt tokens so loss is only on the answer
        prompt_len        = prompt_enc.input_ids.shape[1]
        labels[:prompt_len] = IGNORE_INDEX

        return {
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": full_enc.attention_mask.squeeze(0),
            "images":         pixel_values,
        }


def make_collate_fn(pad_token_id: int):
    """Return a collate function that pads input_ids with the tokenizer's pad token."""
    from torch.nn.utils.rnn import pad_sequence

    def collate_fn(batch):
        # Pad input_ids with the real pad token (eos for Mistral), not <unk>=0.
        # IGNORE_INDEX (-100) is only valid for labels; embed_tokens crashes on negatives.
        # attention_mask=0 for padded positions ensures they don't affect loss.
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


def set_stage1_trainable(model: LlavaMistralXAttnForCausalLM):
    """Stage 1: train cross-attention only."""
    freeze_all(model)
    for p in model.model.cross_attn_modules.parameters():
        p.requires_grad = True
    _print_trainable(model, "Stage 1")


def set_stage2_trainable(
    model: LlavaMistralXAttnForCausalLM,
    top_k_layers: int = 4,
):
    """Stage 2: train cross-attention + top K LLM decoder layers."""
    freeze_all(model)

    # Cross-attention modules
    for p in model.model.cross_attn_modules.parameters():
        p.requires_grad = True

    # Top K decoder layers
    num_layers = len(model.model.layers)
    for i in range(num_layers - top_k_layers, num_layers):
        for p in model.model.layers[i].parameters():
            p.requires_grad = True

    # LM head
    for p in model.lm_head.parameters():
        p.requires_grad = True

    _print_trainable(model, f"Stage 2 (top {top_k_layers} layers)")


def set_stage3_trainable(
    model: LlavaMistralXAttnForCausalLM,
    top_k_layers: int = 4,
):
    """Stage 3: same as Stage 2 (instruction tuning, lower LR)."""
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
    # model weights: checkpoint dir when resuming, base_model otherwise
    model_path = args.resume_from_checkpoint if args.resume_from_checkpoint else args.base_model
    is_resuming = bool(args.resume_from_checkpoint)

    print(f"Loading model from: {model_path}" + (" (resuming checkpoint)" if is_resuming else ""))

    # Tokenizer always loads from base_model, not from the checkpoint dir.
    # Checkpoint dirs saved with save_only_model=True contain only model weights
    # (config.json + model.safetensors) — no tokenizer files (tokenizer.model,
    # tokenizer_config.json, special_tokens_map.json) are present there.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Config: when resuming, hyper-params are already stored in the checkpoint config.
    # When starting fresh, apply the CLI args.
    if is_resuming:
        config = LlavaMistralXAttnConfig.from_pretrained(model_path)
    else:
        config = LlavaMistralXAttnConfig.from_pretrained(
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
        model = LlavaMistralXAttnForCausalLM.from_pretrained(
            model_path,
            config=config,
            quantization_config=bnb_config,
            device_map="auto",
        )
        # Prepares quantized layers for gradient flow; skip grad_ckpt here, we enable it later.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        model = LlavaMistralXAttnForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
        )

    # Load vision tower
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(dtype=torch.bfloat16)
    image_processor = vision_tower.image_processor

    if len(model.model.cross_attn_modules) > 0:
        # cross_attn_modules were pre-built in __init__() using config.cross_attn_vision_dim,
        # and from_pretrained already loaded the saved weights into them.
        # Only re-attach the forward hooks (runtime objects, never serialized).
        model.model.rebuild_hooks()
    else:
        # First-ever build: loading from the original HuggingFace model which has no
        # cross_attn_modules. Build from scratch now that vision dims are known.
        vision_dim  = vision_tower.hidden_size
        num_patches = vision_tower.num_patches   # e.g. 576 for CLIP-ViT-L/14@336
        model.model.build_cross_attention_modules(
            vision_dim=vision_dim, num_patches=num_patches
        )

    # Move each cross-attention module to the same device as the decoder layer.
    # dtype policy:
    #   fp16 AMP (quantized: 8-bit or 4-bit): keep FP32 — fp16 AMP uses GradScaler
    #                             whose unscale_() crashes on non-float32 trainable params.
    #   bf16 AMP (default):      cast to bfloat16 — bf16 AMP has no GradScaler, so
    #                             bfloat16 params are safe and save ~3 GB VRAM
    #                             (17 modules × 41M params × 2 bytes saved on weights + grads).
    ca_train_dtype = torch.float32 if quantized else torch.bfloat16
    layer_index_map = model.model._layer_index_map()
    for layer_idx, module_pos in layer_index_map.items():
        layer_device = next(model.model.layers[layer_idx].parameters()).device
        model.model.cross_attn_modules[module_pos].to(device=layer_device, dtype=ca_train_dtype)

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
    p.add_argument("--output_dir",   type=str,   default="./output")
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
        help="Path to a saved checkpoint directory to resume training from. "
             "Example: ./output/stage1/checkpoint-500",
    )
    p.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load base LLM in 8-bit (bitsandbytes) to save ~7 GB VRAM. "
             "Use on g5.xlarge (24 GB GPU). Cross-attention modules stay in float32.",
    )
    p.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load base LLM in 4-bit NF4 (bitsandbytes) to save more VRAM than "
             "--load_in_8bit (~half the base model's footprint). Needed on 16GB "
             "GPUs (e.g. Kaggle T4) where the 723M fp32 cross-attention modules "
             "plus an 8-bit base model still do not fit. Mutually exclusive with "
             "--load_in_8bit.",
    )
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for cross-attention module init, dropout, and "
                        "data shuffling. Default 42 reproduces the original reported run.")
    p.add_argument("--save_steps", type=int, default=None,
                   help="If set, checkpoint every N optimizer steps instead of every "
                        "epoch — use on time-limited environments (e.g. Kaggle sessions) "
                        "so a mid-epoch timeout does not lose all progress since the last "
                        "epoch boundary. Default (unset) reproduces the original epoch-based "
                        "checkpointing used for the reported AWS run.")
    return p.parse_args()


LR_MAP = {
    1: 1e-5,   # Stage 1: cross-attn only — 5e-5 caused explosion with 17 coupled modules in fp16
    2: 5e-5,   # Stage 2: mixed (LLM layers unfreeze, more stable)
    3: 2e-5,   # Stage 3: instruction, lower LR
}

FREEZE_MAP = {
    1: set_stage1_trainable,
    2: set_stage2_trainable,
    3: set_stage3_trainable,
}


def main():
    args = parse_args()
    quantized = args.load_in_8bit or args.load_in_4bit

    # Must run before model load: the 17 cross-attention modules (Q/K/V, output
    # projection) are freshly initialized (not from the LLaVA-Med checkpoint), so
    # the seed here controls their init in addition to dropout and data shuffling.
    set_seed(args.seed)

    # Reduce CUDA allocator fragmentation — helps when many small tensors are allocated
    # and freed during gradient checkpointing recompute passes.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # ---- model ----
    model, tokenizer, image_processor = load_model_and_tokenizer(args)

    # ---- freeze / unfreeze per stage ----
    freeze_fn = FREEZE_MAP[args.stage]
    if args.stage in (2, 3):
        freeze_fn(model, top_k_layers=args.top_k_layers)
    else:
        freeze_fn(model)

    # Gradient checkpointing: recompute activations during backward instead of
    # storing them — trades ~30% compute for ~40% activation memory savings.
    # Critical on 32 GB GPU with 7B model + multiple cross-attn layers.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ---- dataset ----
    dataset = MedicalVQADataset(
        data_path=args.data_path,
        image_root=args.image_root,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=args.max_length,
    )

    # ---- training arguments ----
    output_dir = os.path.join(args.output_dir, f"stage{args.stage}")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=LR_MAP[args.stage],
        warmup_steps=100,
        lr_scheduler_type="cosine",
        max_grad_norm=0.1,   # stricter: 17 coupled cross-attn modules amplify gradient norm
        bf16=not quantized,    # bf16 AMP only for full-precision mode;
        fp16=quantized,        # quantized model uses fp16 AMP (bnb compute dtype)
        optim="adamw_bnb_8bit",        # 8-bit Adam: optimizer states 0.7 GB thay vì 5.4 GB (fp32)
                                       # tiết kiệm ~4.7 GB VRAM — cần thiết để fit vào A10G 24 GB
        logging_steps=100,   # Kaggle mirrors all stdout into a notebook file on
                              # disk; frequent logging (every 10 steps) filled
                              # /kaggle/working over ~750 steps and crashed
                              # checkpoint saves with "No space left on device".
        save_strategy="steps" if args.save_steps else "epoch",
        save_steps=args.save_steps if args.save_steps else 500,
        save_total_limit=1,            # Trainer writes the new checkpoint before deleting
                                       # ones over the limit, so during that write N+1
                                       # checkpoints briefly coexist on disk. On Kaggle's
                                       # small /kaggle/working quota, limit=2 meant 3
                                       # checkpoints momentarily on disk (~confirmed: crashed
                                       # with "No space left on device" at the 3rd checkpoint,
                                       # step 750, twice in a row regardless of log volume).
                                       # limit=1 caps that transient at 2 checkpoints.
        save_only_model=True,          # không lưu optimizer/scheduler state (~50% nhỏ hơn)
        remove_unused_columns=False,   # important: we have custom "images" column
        dataloader_num_workers=2,      # giảm worker để tránh OOM trên shared memory
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    # ---- trainer ----
    # When quantized (8-bit or 4-bit), Trainer.__init__ refuses to train quantized models
    # unless they are wrapped with PEFT (message: "cannot perform fine-tuning on
    # purely quantized models ..."). The check is inline in Trainer.__init__ (not a
    # separately callable/patchable helper, and its exact line number has moved
    # across transformers versions), and it triggers purely off the standard
    # `model.is_loaded_in_8bit` / `is_loaded_in_4bit` flags set by bitsandbytes.
    # Our cross-attention modules (float32, requires_grad=True) serve exactly the
    # same role as LoRA adapters — the int8 base weights are frozen — so this is a
    # false positive here. Rather than chase the exact check across versions,
    # temporarily hide the flags that trigger it, then restore them immediately
    # after construction so the rest of training (which may rely on them for
    # dtype handling) sees the correct state.
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
                alphas = [m.alpha for m in model.model.cross_attn_modules]
                print(f"  [XAttn step={state.global_step}] "
                      f"alpha={alphas[0]:.4f} (fixed)")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=make_collate_fn(tokenizer.pad_token_id),
        callbacks=[GateMonitor()],
    )

    for _flag, _value in _hidden_quant_flags.items():
        setattr(model, _flag, _value)

    print(f"\n=== Starting Stage {args.stage} training ===")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Strip quantization_config from the saved config.json.
    # When trained with --load_in_8bit/--load_in_4bit, from_pretrained bakes
    # quantization_config into config.json. On reload without quantization this
    # creates Linear8bitLt/Linear4bit layers whose state is never set → inference
    # crashes. The saved weights are already float16 (bnb dequantizes before
    # saving), so removing this key lets future loads use plain bfloat16.
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

    print(f"=== Stage {args.stage} complete. Saved to {output_dir} ===")


if __name__ == "__main__":
    main()
