"""
Chan doan nhanh: kiem tra ranh gioi prompt_len co dung khong trong
train_lora_qwenvl.py - co token nao bi "leak" vao vung tinh loss (sau
prompt_len) ma khong phai la van ban cau tra loi tieng Viet thuc su khong.

Chay song song voi job training dang chay (KHONG dung GPU nhieu, chi load
processor + tokenize, khong load model 7B) - mo 1 tab tmux khac de chay.

Usage:
  python debug_qwenvl_labels.py --base_model Qwen/Qwen2-VL-7B-Instruct \
      --data_path data/dataset_train_minimal.jsonl --image_root ./bone_data --n 3
"""
import argparse
import json
import os

from PIL import Image
from transformers import AutoProcessor

IGNORE_INDEX = -100
FORMAT_HINT = (
    "\n\nTrả lời ngắn gọn bằng tiếng Việt, đúng theo cấu trúc: "
    "\"Kết quả xét nghiệm: X quang. Chẩn đoán xác định: <chẩn đoán>.\" "
    "(hoặc \"Điều trị: <hướng điều trị>.\" nếu câu hỏi yêu cầu điều trị). "
    "Không thêm giải thích dài dòng hay văn bản tiếng Anh."
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2-VL-7B-Instruct")
    p.add_argument("--data_path", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)
    p.add_argument("--n", type=int, default=3)
    args = p.parse_args()

    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(
        args.base_model, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )

    with open(args.data_path, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()][: args.n]

    for i, s in enumerate(samples):
        image = Image.open(os.path.join(args.image_root, s["image"])).convert("RGB")
        question = s["question"]
        answer = s["answer"]

        messages_full = [
            {"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": question + FORMAT_HINT}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        messages_prompt = messages_full[:1]

        text_full = processor.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
        text_prompt = processor.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info(messages_full)
        full_enc = processor(text=[text_full], images=image_inputs, videos=video_inputs, return_tensors="pt")
        prompt_enc = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, return_tensors="pt")

        input_ids = full_enc.input_ids[0]
        prompt_len = prompt_enc.input_ids.shape[1]

        print(f"\n========== SAMPLE {i} ==========")
        print(f"answer (ground truth)      : {answer!r}")
        print(f"prompt_len                 : {prompt_len}")
        print(f"total input_ids length     : {input_ids.shape[0]}")
        print(f"--- 15 token NGAY TRUOC prompt_len (phai la phan cuoi cua prompt/template) ---")
        before = input_ids[max(0, prompt_len - 15):prompt_len]
        print(processor.tokenizer.decode(before))
        print(f"--- 15 token NGAY TU prompt_len (PHAI la dau cau tra loi tieng Viet, KHONG phai token anh/template) ---")
        after = input_ids[prompt_len:prompt_len + 15]
        print(repr(processor.tokenizer.decode(after)))
        print(f"--- toan bo vung se tinh loss (tu prompt_len den het) ---")
        full_label_region = input_ids[prompt_len:]
        print(repr(processor.tokenizer.decode(full_label_region)))
        print(f"so sanh voi answer goc: {'KHOP' if answer.strip() in processor.tokenizer.decode(full_label_region) else '!!! KHONG KHOP - CO BUG !!!'}")


if __name__ == "__main__":
    main()
