"""
RAG inference script — so sánh Pre-RAG vs Post-RAG vs No-RAG.

RAG types
---------
  none      : chạy model gốc không có RAG (baseline để so sánh)
  pre       : retrieve top-k similar training cases → prepend vào prompt → generate
  post      : generate initial answer → retrieve top-k cases tương tự initial answer
              → regenerate với augmented context

Retrieval
---------
  - Pre-RAG  : truy vấn bằng question embedding, tìm ca có câu hỏi tương tự
  - Post-RAG : truy vấn bằng initial answer embedding, tìm ca có chẩn đoán tương tự

Usage
-----
# Bước 1 — build index (chỉ chạy 1 lần)
python build_rag_index.py \
    --data_path data_enhanced/dataset_train.jsonl \
    --output_dir rag_index/

# Bước 2 — inference

# No-RAG (baseline so sánh)
python run_rag_inference.py \
    --rag_type   none \
    --model_path ./outputs_enhanced/stage1/stage1 \
    --data_path  data_enhanced/dataset_test_minimal.jsonl \
    --image_root ./xray_data \
    --output_file rag_none_outputs.jsonl

# Pre-RAG
python run_rag_inference.py \
    --rag_type   pre \
    --top_k      3 \
    --model_path ./outputs_enhanced/stage1/stage1 \
    --index_dir  rag_index/ \
    --data_path  data_enhanced/dataset_test_minimal.jsonl \
    --image_root ./xray_data \
    --output_file rag_pre_outputs.jsonl

# Post-RAG
python run_rag_inference.py \
    --rag_type   post \
    --top_k      3 \
    --model_path ./outputs_enhanced/stage1/stage1 \
    --index_dir  rag_index/ \
    --data_path  data_enhanced/dataset_test_minimal.jsonl \
    --image_root ./xray_data \
    --output_file rag_post_outputs.jsonl
"""

import argparse
import json
import os

import pickle
import re

import numpy as np
import torch
from PIL import Image
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoTokenizer

from llava.constants import DEFAULT_IMAGE_TOKEN


# ---------------------------------------------------------------------------
# Model loading (XAttn)
# ---------------------------------------------------------------------------

def load_xattn_model(model_path: str, base_model: str):
    from llava.model import LlavaMistralXAttnForCausalLM, LlavaMistralXAttnConfig

    config = LlavaMistralXAttnConfig.from_pretrained(model_path)
    model = LlavaMistralXAttnForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    llm_device = next(model.model.embed_tokens.parameters()).device
    vision_tower.to(device=llm_device, dtype=torch.bfloat16)
    image_processor = vision_tower.image_processor

    model.model.rebuild_hooks()
    for module in model.model.cross_attn_modules:
        module.to(dtype=torch.bfloat16)

    model.eval()
    return tokenizer, model, image_processor


# ---------------------------------------------------------------------------
# Single inference call
# ---------------------------------------------------------------------------

@torch.inference_mode()
def infer(model, tokenizer, image_processor, image_path: str, question: str,
          max_new_tokens: int = 256) -> str:
    """Mirror of infer_xattn() in run_inference.py — must use the same format."""
    prompt = f"{DEFAULT_IMAGE_TOKEN}\n{question}\nAnswer:"
    input_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).input_ids.cuda()

    image = Image.open(image_path).convert("RGB")
    pixel_values = (
        image_processor(images=image, return_tensors="pt")
        .pixel_values.to(dtype=torch.bfloat16).cuda()
    )

    output_ids = model.generate(
        inputs=input_ids,
        images=pixel_values,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if "Answer:" in decoded:
        decoded = decoded.split("Answer:")[-1].strip()
    return decoded


# ---------------------------------------------------------------------------
# RAG retriever
# ---------------------------------------------------------------------------

def _tokenize_vi(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


class RAGRetriever:
    """BM25 text-based retriever (query by question or answer text)."""

    def __init__(self, index_dir: str):
        with open(os.path.join(index_dir, "q_bm25.pkl"), "rb") as f:
            self.q_bm25 = pickle.load(f)
        with open(os.path.join(index_dir, "a_bm25.pkl"), "rb") as f:
            self.a_bm25 = pickle.load(f)

        self.metadata = []
        with open(os.path.join(index_dir, "metadata.jsonl"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.metadata.append(json.loads(line))

        print(f"[RAG] BM25 index loaded: {len(self.metadata)} training samples")

    def retrieve_by_question(self, query: str, top_k: int):
        tokens = _tokenize_vi(query)
        scores = self.q_bm25.get_scores(tokens)
        idxs = scores.argsort()[-top_k:][::-1]
        return [self.metadata[i] for i in idxs], scores[idxs].tolist()

    def retrieve_by_answer(self, query: str, top_k: int):
        tokens = _tokenize_vi(query)
        scores = self.a_bm25.get_scores(tokens)
        idxs = scores.argsort()[-top_k:][::-1]
        return [self.metadata[i] for i in idxs], scores[idxs].tolist()


class CLIPRetriever:
    """Image-based retriever using CLIP visual embeddings."""

    def __init__(self, index_dir: str):
        import json as _json
        cfg = _json.load(open(os.path.join(index_dir, "config.json")))
        clip_model = cfg["clip_model"]

        print(f"[CLIP] Loading vision encoder: {clip_model}")
        from transformers import CLIPImageProcessor, CLIPVisionModel
        self.processor = CLIPImageProcessor.from_pretrained(clip_model)
        self.clip = CLIPVisionModel.from_pretrained(
            clip_model, torch_dtype=torch.float16
        ).cuda().eval()

        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))

        self.metadata = []
        with open(os.path.join(index_dir, "metadata.jsonl"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.metadata.append(json.loads(line))

        print(f"[CLIP] Index loaded: {self.embeddings.shape[0]} training images")

    @torch.inference_mode()
    def retrieve(self, image_path: str, top_k: int):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to("cuda")
        outputs = self.clip(**inputs)
        q_emb = outputs.pooler_output[0].float().cpu().numpy()
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)

        sims = self.embeddings @ q_emb
        idxs = sims.argsort()[-top_k:][::-1]
        return [self.metadata[i] for i in idxs], sims[idxs].tolist()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_pre_rag_question(original_question: str, retrieved: list) -> str:
    """Append retrieved diagnoses as hints after the original question.

    Keeping retrieved context SHORT (answers only, no question excerpts) to avoid
    exceeding the model's trained max_length=512. Original question comes first so
    the model sees the full task before the hints.
    """
    cases = "\n".join([f"- {c['answer']}" for c in retrieved])
    return f"{original_question}\n\nChẩn đoán tham khảo từ ca tương tự:\n{cases}"


def build_post_rag_question(original_question: str, initial_answer: str,
                             retrieved: list) -> str:
    """Refine prompt using retrieved cases similar to the initial answer.

    Keep context short: only the retrieved answers (no question excerpts)
    to stay within the model's trained max_length=512.
    """
    cases = "\n".join([f"- {c['answer']}" for c in retrieved])
    return (
        f"{original_question}\n\n"
        f"Chẩn đoán tham khảo từ ca tương tự:\n{cases}\n\n"
        f"Dựa trên hình ảnh X-quang và tham khảo trên, đưa ra chẩn đoán xác định."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rag_type",    required=True, choices=["none", "pre", "post"])
    p.add_argument("--model_path",  required=True)
    p.add_argument("--base_model",  default="microsoft/llava-med-v1.5-mistral-7b")
    p.add_argument("--data_path",   required=True)
    p.add_argument("--image_root",  required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--index_dir",      default="rag_index/",
                   help="Required for pre and post RAG")
    p.add_argument("--retrieval_type", default="bm25", choices=["bm25", "clip"],
                   help="bm25: text-based retrieval | clip: image-based retrieval")
    p.add_argument("--clip_index_dir", default="rag_clip_index/",
                   help="CLIP index dir (used when --retrieval_type clip)")
    p.add_argument("--top_k",          type=int, default=3,
                   help="Number of cases to retrieve")
    p.add_argument("--max_new_tokens", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Load model ----
    print(f"Loading XAttn model from {args.model_path}...")
    tokenizer, model, image_processor = load_xattn_model(args.model_path, args.base_model)

    # ---- Load retriever (if needed) ----
    retriever = None
    clip_retriever = None
    if args.rag_type in ("pre", "post"):
        if args.retrieval_type == "clip":
            clip_retriever = CLIPRetriever(args.clip_index_dir)
        else:
            retriever = RAGRetriever(args.index_dir)

    # ---- Load test data ----
    samples = []
    with open(args.data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Test samples: {len(samples)}")

    # ---- Inference ----
    results = []
    for sample in tqdm(samples, desc=f"RAG={args.rag_type}"):
        image_path = os.path.join(args.image_root, sample["image"])
        original_q  = sample["question"]
        reference   = sample.get("answer", "")

        if args.rag_type == "none":
            prediction = infer(model, tokenizer, image_processor,
                               image_path, original_q, args.max_new_tokens)
            retrieved_answers = []

        elif args.rag_type == "pre":
            if clip_retriever:
                retrieved, scores = clip_retriever.retrieve(image_path, args.top_k)
            else:
                retrieved, scores = retriever.retrieve_by_question(original_q, args.top_k)
            augmented_q = build_pre_rag_question(original_q, retrieved)
            prediction  = infer(model, tokenizer, image_processor,
                                image_path, augmented_q, args.max_new_tokens)
            retrieved_answers = [r["answer"] for r in retrieved]

        elif args.rag_type == "post":
            initial_answer = infer(model, tokenizer, image_processor,
                                   image_path, original_q, args.max_new_tokens)
            if clip_retriever:
                retrieved, scores = clip_retriever.retrieve(image_path, args.top_k)
            else:
                retrieved, scores = retriever.retrieve_by_answer(initial_answer, args.top_k)
            augmented_q = build_post_rag_question(original_q, initial_answer, retrieved)
            prediction  = infer(model, tokenizer, image_processor,
                                image_path, augmented_q, args.max_new_tokens)
            retrieved_answers = [r["answer"] for r in retrieved]

        results.append({
            "image":      sample["image"],
            "question":   original_q,
            "reference":  reference,
            "prediction": prediction,
            "rag_type":   args.rag_type,
            "retrieved":  retrieved_answers,
        })

    # ---- Save ----
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    empty = sum(1 for r in results if not r["prediction"].strip())
    print(f"\nDone. {len(results)} samples → {args.output_file}")
    print(f"Empty predictions: {empty}/{len(results)}")


if __name__ == "__main__":
    main()
