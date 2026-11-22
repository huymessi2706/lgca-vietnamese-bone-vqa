"""
Build CLIP image embedding index from training data for image-based RAG.

Encodes each training image with openai/clip-vit-large-patch14-336 (the same
vision encoder used by LLaVA-Med). Retrieval by visual similarity finds cases
with radiographically similar X-rays, which is more relevant than BM25 text
matching on short minimal-context queries.

Usage:
    python build_clip_index.py \
        --data_path  data_enhanced/dataset_train_minimal.jsonl \
        --image_root ./xray_data \
        --output_dir rag_clip_index/
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel


@torch.inference_mode()
def encode_images(image_paths: list, processor, model, batch_size: int, device: str) -> np.ndarray:
    all_embeddings = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # pooler_output = CLS token passed through projection layer — global image repr
        embeddings = outputs.pooler_output  # [B, 768 or 1024]
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # L2 normalize
        all_embeddings.append(embeddings.cpu().float().numpy())
    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  default="data_enhanced/dataset_train_minimal.jsonl")
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", default="rag_clip_index/")
    parser.add_argument("--clip_model", default="openai/clip-vit-large-patch14-336",
                        help="Already cached on server as LLaVA-Med vision tower")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load data ----
    samples = []
    with open(args.data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # Filter samples with missing images
    valid = []
    missing = 0
    for s in samples:
        p = os.path.join(args.image_root, s["image"])
        if os.path.exists(p):
            valid.append(s)
        else:
            missing += 1
    if missing:
        print(f"Skipped {missing} samples with missing images")
    print(f"Encoding {len(valid)} training images")

    # ---- Load CLIP ----
    print(f"Loading CLIP: {args.clip_model}")
    processor = CLIPImageProcessor.from_pretrained(args.clip_model)
    model = CLIPVisionModel.from_pretrained(args.clip_model, torch_dtype=torch.float16)
    model = model.to(device).eval()

    # ---- Encode ----
    image_paths = [os.path.join(args.image_root, s["image"]) for s in valid]
    embeddings = encode_images(image_paths, processor, model, args.batch_size, device)
    print(f"Embeddings shape: {embeddings.shape}")

    # ---- Save ----
    np.save(os.path.join(args.output_dir, "embeddings.npy"), embeddings)

    meta_path = os.path.join(args.output_dir, "metadata.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for s in valid:
            f.write(json.dumps({
                "question": s["question"],
                "answer":   s["answer"],
                "image":    s["image"],
            }, ensure_ascii=False) + "\n")

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({"clip_model": args.clip_model, "n_samples": len(valid),
                   "dim": embeddings.shape[1]}, f, indent=2)

    print(f"\nDone. Index saved → {args.output_dir}")
    print(f"  embeddings.npy  shape={embeddings.shape}")
    print(f"  metadata.jsonl  ({len(valid)} records)")

    # Sanity check: self-retrieval — top-1 of first sample should be itself
    q = embeddings[0:1]
    sims = (embeddings @ q.T).squeeze()
    top_idx = sims.argsort()[-3:][::-1]
    print("\n--- Sanity check: top-3 similar to sample 0 ---")
    for i in top_idx:
        print(f"  [{sims[i]:.4f}] {valid[i]['answer'][:80]}")


if __name__ == "__main__":
    main()
