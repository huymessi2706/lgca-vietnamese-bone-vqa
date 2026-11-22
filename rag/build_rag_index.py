"""
Build BM25 retrieval index from training data for RAG experiments.

BM25 (Best Matching 25) là lexical retrieval — không cần neural embedding,
không có dependency conflict. Hoạt động tốt với tiếng Việt y tế vì
thuật ngữ y khoa thường là từ ghép đặc thù (thoát vị, thoái hóa, hẹp ống sống...).

Usage:
    python build_rag_index.py \
        --data_path data_enhanced/dataset_train_minimal.jsonl \
        --output_dir rag_index/
"""

import argparse
import json
import os
import pickle
import re

from rank_bm25 import BM25Okapi


def tokenize_vi(text: str) -> list[str]:
    """Simple Vietnamese tokenizer: lowercase + split on whitespace/punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  default="data_enhanced/dataset_train_minimal.jsonl")
    parser.add_argument("--output_dir", default="rag_index/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load data ----
    samples = []
    with open(args.data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} training samples")

    # ---- Tokenize ----
    q_corpus = [tokenize_vi(s["question"]) for s in samples]
    a_corpus = [tokenize_vi(s["answer"])   for s in samples]

    # ---- Build BM25 indices ----
    print("Building BM25 question index...")
    q_bm25 = BM25Okapi(q_corpus)

    print("Building BM25 answer index (for Post-RAG)...")
    a_bm25 = BM25Okapi(a_corpus)

    # ---- Save ----
    with open(os.path.join(args.output_dir, "q_bm25.pkl"), "wb") as f:
        pickle.dump(q_bm25, f)
    with open(os.path.join(args.output_dir, "a_bm25.pkl"), "wb") as f:
        pickle.dump(a_bm25, f)

    meta_path = os.path.join(args.output_dir, "metadata.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({
                "question": s["question"],
                "answer":   s["answer"],
                "image":    s["image"],
            }, ensure_ascii=False) + "\n")

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({"type": "bm25", "n_samples": len(samples)}, f, indent=2)

    print(f"\nDone. Index saved → {args.output_dir}")
    print(f"  q_bm25.pkl   (question index, {len(samples)} docs)")
    print(f"  a_bm25.pkl   (answer index,   {len(samples)} docs)")
    print(f"  metadata.jsonl")

    # Quick sanity check
    test_query = tokenize_vi("thoái hóa cột sống thắt lưng")
    scores = q_bm25.get_scores(test_query)
    top_idx = scores.argsort()[-3:][::-1]
    print("\n--- Sanity check: top-3 for 'thoái hóa cột sống thắt lưng' ---")
    for i in top_idx:
        print(f"  [{scores[i]:.2f}] {samples[i]['answer'][:80]}")


if __name__ == "__main__":
    main()
