"""
Pre-augment training data với BM25 retrieved context cho RAG-aware training.

Với mỗi training sample, retrieve top-k ca tương tự (loại trừ chính nó),
rồi nối retrieved diagnoses vào câu hỏi — cùng format với inference.

Model được train với format này sẽ học cách sử dụng retrieved context
thay vì bị confuse khi gặp nó lúc inference.

Usage:
    python build_rag_train_data.py \
        --data_path  data_enhanced/dataset_train_minimal.jsonl \
        --index_dir  rag_index/ \
        --output     data_enhanced/dataset_train_minimal_rag.jsonl \
        --top_k      3 \
        --aug_ratio  1.0
"""

import argparse
import json
import os
import pickle
import re


def tokenize_vi(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def retrieve_excluding_self(bm25, metadata, query_tokens, self_answer: str, top_k: int):
    """Retrieve top-k, excluding the sample itself (identified by exact answer match)."""
    scores = bm25.get_scores(query_tokens)
    ranked = scores.argsort()[::-1]

    results = []
    for idx in ranked:
        if metadata[idx]["answer"].strip() == self_answer.strip():
            continue  # skip self
        results.append(metadata[idx])
        if len(results) == top_k:
            break
    return results


def build_augmented_question(original_question: str, retrieved: list) -> str:
    cases = "\n".join([f"- {c['answer']}" for c in retrieved])
    return f"{original_question}\n\nChẩn đoán tham khảo từ ca tương tự:\n{cases}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  default="data_enhanced/dataset_train_minimal.jsonl")
    parser.add_argument("--index_dir",  default="rag_index/")
    parser.add_argument("--output",     default="data_enhanced/dataset_train_minimal_rag.jsonl")
    parser.add_argument("--top_k",      type=int, default=3)
    parser.add_argument("--aug_ratio",  type=float, default=1.0,
                        help="Fraction of samples to augment (1.0 = all). "
                             "Remaining samples kept as-is for robustness.")
    args = parser.parse_args()

    # ---- Load data ----
    samples = []
    with open(args.data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} training samples")

    # ---- Load BM25 index ----
    with open(os.path.join(args.index_dir, "q_bm25.pkl"), "rb") as f:
        q_bm25 = pickle.load(f)

    metadata = []
    with open(os.path.join(args.index_dir, "metadata.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))
    print(f"BM25 index: {len(metadata)} docs")

    # ---- Augment ----
    import random
    random.seed(42)

    augmented = 0
    kept_original = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for s in samples:
            if random.random() > args.aug_ratio:
                # Keep original (no RAG) — helps model work without retrieved context too
                out.write(json.dumps(s, ensure_ascii=False) + "\n")
                kept_original += 1
                continue

            query_tokens = tokenize_vi(s["question"])
            retrieved = retrieve_excluding_self(q_bm25, metadata, query_tokens,
                                                self_answer=s["answer"], top_k=args.top_k)

            if not retrieved:
                # Fallback: no retrieved cases available
                out.write(json.dumps(s, ensure_ascii=False) + "\n")
                kept_original += 1
                continue

            record = dict(s)
            record["question"] = build_augmented_question(s["question"], retrieved)
            record["original_question"] = s["question"]
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            augmented += 1

    total = augmented + kept_original
    print(f"\nDone → {args.output}")
    print(f"  Augmented (with RAG):  {augmented}/{total} ({100*augmented/total:.1f}%)")
    print(f"  Original (no RAG):     {kept_original}/{total} ({100*kept_original/total:.1f}%)")

    # Preview 2 augmented samples
    print("\n--- Preview ---")
    with open(args.output, encoding="utf-8") as f:
        count = 0
        for line in f:
            s = json.loads(line)
            if "original_question" in s:
                print(f"\nQuestion:\n{s['question'][:400]}")
                print(f"Answer: {s['answer'][:100]}")
                count += 1
            if count >= 2:
                break


if __name__ == "__main__":
    main()
