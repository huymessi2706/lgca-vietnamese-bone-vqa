"""
Run inference with the trained Qwen2-VL + XAttn model, in the same JSONL
output format as run_inference.py (so compute_metrics.py works unchanged).

Usage:
  python run_inference_qwenvl_xattn.py \
      --model_path  ./outputs_qwenvl_xattn_minimal \
      --base_model  Qwen/Qwen2-VL-7B-Instruct \
      --data_path   bone_data/dataset_test_minimal.jsonl \
      --image_root  ./bone_data \
      --output_file results/outputs/qwenvl_xattn_minimal_outputs.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from dataset import QwenVLXAttnDataset
from modeling_qwen2vl_xattn import Qwen2VLXAttnForCausalLM


@torch.inference_mode()
def infer_one(model, processor, image_path: str, question: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": question}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = processor.tokenizer(text_prompt, return_tensors="pt").input_ids.cuda()

    image = Image.open(image_path).convert("RGB")
    image_enc = processor.image_processor(images=image, return_tensors="pt")
    pixel_values = image_enc.pixel_values.to(dtype=torch.bfloat16).cuda()
    image_grid_thw = image_enc.image_grid_thw.cuda()

    output_ids = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    generated_ids = output_ids[:, input_ids.shape[1]:]
    answer = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    return answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen2-VL-7B-Instruct")
    p.add_argument("--data_path", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    print(f"Loading model from {args.model_path} ...")
    model = Qwen2VLXAttnForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
    )
    model.rebuild_hooks_after_load()
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )

    with open(args.data_path, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]

    results = []
    for i, sample in enumerate(tqdm(samples)):
        image_path = os.path.join(args.image_root, sample["image"])
        if not os.path.exists(image_path):
            continue
        try:
            prediction = infer_one(model, processor, image_path, sample["question"], args.max_new_tokens)
        except Exception as e:
            print(f"[Error] sample {i}: {e}")
            prediction = ""
        results.append({
            "question_id": i,
            "image": sample["image"],
            "question": sample["question"],
            "reference": sample.get("answer", ""),
            "prediction": prediction,
            "model_type": "qwenvl_xattn",
            "domain": sample.get("domain", {}),
            "type": sample.get("type", ""),
        })

    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} predictions -> {args.output_file}")


if __name__ == "__main__":
    main()
