"""
LoRA fine-tuning Qwen2-VL tren tap du lieu X-quang xuong tieng Viet
(minimal-context) - de kiem dinh gia thuyet: liem the manh full-context cua
Qwen-VL zero-shot la do doc text tot, khong phai do doc anh X-quang gioi;
va de xem sau khi fine-tune, Qwen-VL co dong kip XAttn o minimal-context
khong (THESIS_REVISION_PLAN.md).

KHONG dung chung env "llava" - chay trong env "qwenvl" rieng (xem
run_inference_qwenvl.py de biet cach tao env).
Can them: pip install peft bitsandbytes qwen-vl-utils

Usage:
  # 1. Tao tap train minimal-context (dung lai script da co cho test set)
  python build_minimal_test.py \
      --input  data/dataset_train.jsonl \
      --output data/dataset_train_minimal.jsonl

  # 2. Train
  python train_lora_qwenvl.py \
      --base_model Qwen/Qwen2-VL-7B-Instruct \
      --data_path  data/dataset_train_minimal.jsonl \
      --image_root ./bone_data \
      --output_dir ./outputs_v2/lora_qwenvl_minimal \
      --num_epochs 3 --lora_r 16

  # 3. Infer + eval bang dung run_inference_qwenvl.py (model_path = output_dir,
  #    dataset_test_minimal.jsonl), roi compute_metrics.py + proxy_classification_metrics.py
  #    nhu da lam voi Qwen-VL zero-shot.
"""

import argparse
import json
import os

import torch
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Trainer, TrainingArguments

IGNORE_INDEX = -100

# Giong FORMAT_HINT trong run_inference_qwenvl.py - giu nhat quan train/infer
FORMAT_HINT = (
    "\n\nTrả lời ngắn gọn bằng tiếng Việt, đúng theo cấu trúc: "
    "\"Kết quả xét nghiệm: X quang. Chẩn đoán xác định: <chẩn đoán>.\" "
    "(hoặc \"Điều trị: <hướng điều trị>.\" nếu câu hỏi yêu cầu điều trị). "
    "Không thêm giải thích dài dòng hay văn bản tiếng Anh."
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class QwenVLVQADataset(Dataset):
    def __init__(self, data_path, image_root, processor, max_length=1024):
        with open(data_path, encoding="utf-8") as f:
            all_samples = [json.loads(l) for l in f if l.strip()]

        self.image_root = image_root
        self.processor = processor
        self.max_length = max_length

        valid = []
        skipped_img = 0
        for s in all_samples:
            if not os.path.exists(os.path.join(image_root, s["image"])):
                skipped_img += 1
                continue
            valid.append(s)
        self.samples = valid
        if skipped_img:
            print(f"[Dataset] Skipped {skipped_img} samples with missing images")
        print(f"[Dataset] {len(self.samples)} samples ready")

    def __len__(self):
        return len(self.samples)

    def _build_messages(self, image, question, answer=None):
        user_msg = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question + FORMAT_HINT},
            ],
        }
        if answer is None:
            return [user_msg]
        return [user_msg, {"role": "assistant", "content": [{"type": "text", "text": answer}]}]

    def __getitem__(self, idx):
        from qwen_vl_utils import process_vision_info

        sample = self.samples[idx]
        question = sample["question"]
        answer = sample["answer"]
        image = Image.open(os.path.join(self.image_root, sample["image"])).convert("RGB")

        messages_full = self._build_messages(image, question, answer)
        messages_prompt = self._build_messages(image, question, answer=None)

        text_full = self.processor.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
        text_prompt = self.processor.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info(messages_full)

        full_enc = self.processor(
            text=[text_full], images=image_inputs, videos=video_inputs,
            padding=False, truncation=True, max_length=self.max_length, return_tensors="pt",
        )
        prompt_enc = self.processor(
            text=[text_prompt], images=image_inputs, videos=video_inputs,
            padding=False, truncation=True, max_length=self.max_length, return_tensors="pt",
        )

        input_ids = full_enc.input_ids[0]
        prompt_len = prompt_enc.input_ids.shape[1]
        prompt_len = min(prompt_len, input_ids.shape[0])  # phong truong hop truncation cat ngan hon prompt

        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX

        item = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": full_enc.attention_mask[0],
            "pixel_values": full_enc.pixel_values,
            "image_grid_thw": full_enc.image_grid_thw,
        }
        # mm_token_type_ids: bat buoc voi transformers ban moi de tinh M-RoPE
        # dung cho input co anh. Khong phai moi ban transformers deu tra ve
        # field nay nen kiem tra truoc khi lay.
        if hasattr(full_enc, "mm_token_type_ids"):
            item["mm_token_type_ids"] = full_enc.mm_token_type_ids[0]
        return item


def make_collate_fn(pad_token_id: int):
    def collate_fn(batch):
        # batch_size=1 -> khong can pad phuc tap, nhung Trainer van goi collate_fn voi list
        input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
        pixel_values = torch.cat([b["pixel_values"] for b in batch], dim=0)
        image_grid_thw = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
        out = {
            "input_ids": input_ids, "labels": labels, "attention_mask": attention_mask,
            "pixel_values": pixel_values, "image_grid_thw": image_grid_thw,
        }
        if "mm_token_type_ids" in batch[0]:
            out["mm_token_type_ids"] = pad_sequence(
                [b["mm_token_type_ids"] for b in batch], batch_first=True, padding_value=0
            )
        return out
    return collate_fn


class NaNSafeTrainer(Trainer):
    """Bo qua step co loss NaN/Inf thay vi lam hong gradient ca batch (giong train_lora_baseline.py)."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
            loss = torch.tensor(0.0, requires_grad=True, device=loss.device)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2-VL-7B-Instruct")
    p.add_argument("--data_path", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--output_dir", default="./outputs_v2/lora_qwenvl_minimal")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--max_length", type=int, default=1536)
    p.add_argument("--min_pixels", type=int, default=256 * 28 * 28,
                   help="Gioi han duoi so pixel anh dua vao processor -> bound so visual token (~256 token)")
    p.add_argument("--max_pixels", type=int, default=1024 * 28 * 28,
                   help="Gioi han tren so pixel anh -> bound so visual token (~1024 token). "
                        "Anh X-quang goc co the sinh >2000 token neu khong gioi han, gay loi truncation.")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"Loading base model: {args.base_model}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
    )
    processor = AutoProcessor.from_pretrained(
        args.base_model, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )
    print(f"[Processor] min_pixels={args.min_pixels} max_pixels={args.max_pixels} "
          f"(~{args.min_pixels // (28*28)}-{args.max_pixels // (28*28)} visual tokens/anh)")
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dong bang toan bo, chi LoRA tren cac proj cua language model.
    # Vision tower (Qwen2-VL ViT) dung ten module khac ("qkv", "proj"...) nen
    # KHONG bi LoRA-hoa boi target_modules duoi day - giong cach train_lora_baseline.py
    # tu dong bo qua CLIP vision tower cua LLaVA-Med.
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

    dataset = QwenVLVQADataset(
        data_path=args.data_path, image_root=args.image_root,
        processor=processor, max_length=args.max_length,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=True,
        optim="adamw_bnb_8bit",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
    )

    trainer = NaNSafeTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=make_collate_fn(tokenizer.pad_token_id),
    )

    print(f"\n=== LoRA Qwen2-VL training ({args.num_epochs} epochs, r={args.lora_r}) ===")
    trainer.train()

    print("\nMerging LoRA weights into base model ...")
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"=== Done. Merged model saved -> {args.output_dir} ===")


if __name__ == "__main__":
    main()
