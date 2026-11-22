"""
Zero-shot inference with Qwen2-VL — external baseline for thesis comparison
(THESIS_REVISION_PLAN.md, mục #4b).

KHÔNG dùng chung conda env với LLaVA-Med (env "llava") — Qwen2-VL cần
transformers>=4.46, xung đột với transformers==4.36.2 đang pin cho LLaVA-Med.
Tạo env riêng trước khi chạy file này:

    conda create -n qwenvl python=3.10 -y
    conda activate qwenvl
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install "transformers>=4.46.0" accelerate qwen-vl-utils pillow

Usage
-----
python run_inference_qwenvl.py \\
    --model_path  Qwen/Qwen2-VL-7B-Instruct \\
    --data_path   data/dataset_test.jsonl \\
    --image_root  ./bone_data \\
    --output_file results/qwenvl_outputs.jsonl

Dùng model nhỏ hơn (nhanh/rẻ hơn, nếu VRAM/thời gian hạn chế):
    --model_path Qwen/Qwen2-VL-2B-Instruct

Output JSONL có cùng schema với run_inference.py (question_id, image,
question, reference, prediction, model_type) — nạp trực tiếp vào
compute_metrics.py cùng với các file kết quả khác:

python compute_metrics.py \\
    --files results/baseline_outputs.jsonl results/xattn_minimal_trained_outputs.jsonl results/qwenvl_outputs.jsonl \\
    --names Baseline XAttn QwenVL-zeroshot \\
    --output_file results/metrics_with_qwenvl.json
"""

import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data loading — same custom bone-data JSONL format as run_inference.py
# ---------------------------------------------------------------------------

def parse_sample(line: dict) -> tuple:
    """Returns (question_id, image_filename, question_text, reference_answer)."""
    question = line.get("question", "")
    reference = line.get("answer", line.get("reference", ""))
    qid = line.get("question_id", line.get("id", 0))
    return qid, line["image"], question, reference


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_qwenvl(model_path: str, min_pixels: int = None, max_pixels: int = None):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print(f"[Qwen-VL] Loading {model_path} ...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor_kwargs = {}
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
    if processor_kwargs:
        print(f"[Qwen-VL] Processor pixel bounds: {processor_kwargs} "
              f"(phai khop voi luc train_lora_qwenvl.py de nhat quan do phan giai)")
    model.eval()
    print("[Qwen-VL] Ready.")
    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

# Qwen-VL không được train trên format câu trả lời của dataset này
# ("Kết quả xét nghiệm: ... Chẩn đoán xác định: ..."). Không có hint này,
# BLEU/ROUGE (đo trùng khớp từ vựng) sẽ dìm điểm Qwen-VL chỉ vì khác văn
# phong, dù nội dung y khoa có thể đúng -- không phản ánh đúng năng lực
# model. Đây là prompt engineering hợp lệ cho so sánh zero-shot, không
# phải fine-tune, nên không vi phạm tính "zero-shot" của baseline này.
FORMAT_HINT = (
    "\n\nTrả lời ngắn gọn bằng tiếng Việt, đúng theo cấu trúc: "
    "\"Kết quả xét nghiệm: X quang. Chẩn đoán xác định: <chẩn đoán>.\" "
    "(hoặc \"Điều trị: <hướng điều trị>.\" nếu câu hỏi yêu cầu điều trị). "
    "Không thêm giải thích dài dòng hay văn bản tiếng Anh."
)


@torch.inference_mode()
def infer_qwenvl(model, processor, image_path: str, question: str, max_new_tokens: int) -> str:
    from qwen_vl_utils import process_vision_info

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question + FORMAT_HINT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Zero-shot Qwen2-VL inference for thesis comparison")
    parser.add_argument("--model_path", default="Qwen/Qwen2-VL-7B-Instruct",
                        help="HF model ID. Use Qwen/Qwen2-VL-2B-Instruct for a smaller/faster run.")
    parser.add_argument("--data_path", required=True,
                        help="Same dataset_test.jsonl used for baseline/XAttn/LoRA runs")
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--min_pixels", type=int, default=None,
                        help="Chi can set khi infer model da train bang train_lora_qwenvl.py "
                             "-- phai khop voi --min_pixels luc train. Bo trong (None) cho "
                             "zero-shot Qwen-VL goc (dung do phan giai mac dinh).")
    parser.add_argument("--max_pixels", type=int, default=None,
                        help="Tuong tu min_pixels, phai khop voi luc train.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    model, processor = load_qwenvl(args.model_path, args.min_pixels, args.max_pixels)

    with open(args.data_path, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(samples)} test samples from {args.data_path}")

    # QUAN TRỌNG: compute_metrics.py ghép các file theo CHỈ SỐ DÒNG (không theo
    # question_id), nên ta KHÔNG được skip-và-bỏ-dòng khi gặp lỗi/ảnh thiếu —
    # nếu không, dòng N của file này sẽ bị so sánh với reference của dòng N
    # nhưng prediction lại tương ứng với sample N+1 (lệch hàng âm thầm, không
    # có cảnh báo). Luôn ghi đủ len(samples) dòng, theo đúng thứ tự.
    missing_img, errors = 0, 0
    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for sample in tqdm(samples, desc="Inferring (qwenvl)"):
            qid, image_file, question, reference = parse_sample(sample)
            image_path = os.path.join(args.image_root, image_file)

            if not os.path.exists(image_path):
                missing_img += 1
                prediction = ""
            else:
                try:
                    prediction = infer_qwenvl(model, processor, image_path, question, args.max_new_tokens)
                except Exception as e:
                    print(f"\n  [ERROR] qid={qid}: {e}")
                    prediction = ""
                    errors += 1

            record = {
                "question_id": qid,
                "image": image_file,
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "model_type": "qwenvl",
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    total = len(samples)
    print(f"\nDone. Wrote {total} lines (preserves alignment with other model output files).")
    print(f"  {total - missing_img - errors}/{total} had a real generated prediction.")
    if missing_img:
        print(f"  {missing_img} samples had missing image -> written with prediction=''")
    if errors:
        print(f"  Errors  {errors} samples (see messages above)")
    print(f"Results saved -> {args.output_file}")


if __name__ == "__main__":
    main()
