"""
Pre-augment training data with BM25 retrieved context for RAG-aware training,
using a STRICTER exclusion filter than build_rag_train_data.py.

In addition to excluding the exact-answer-match self-sample, this version
also excludes candidates that are:
  - same-patient (same ma_ho_so as the query), or
  - near-duplicate-answer (token Jaccard similarity >= 0.8 to the query's
    own answer, computed on the same normalized tokens used for BM25).

This directly tests whether the residual retrieval contamination
quantified in Section "Discussion" (6.19% same-patient, 4.22%
near-duplicate-answer out of 20,421 retrieved train pairs) measurably
affects RAG results once excluded.

Usage:
    python build_rag_train_data_strict.py \
        --data_path  data/dataset_train.jsonl \
        --index_dir  rag_index_strict/ \
        --output     data/dataset_train_rag_strict.jsonl \
        --top_k      3 \
        --aug_ratio  1.0
"""

import argparse
import json
import os
import pickle
import re


def tokenize_vi(text: str) -> list:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def jaccard(a_tokens, b_tokens):
    sa, sb = set(a_tokens), set(b_tokens)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def retrieve_excluding_self_strict(bm25, metadata, query_tokens, self_answer: str,
                                    self_ma_ho_so: str, top_k: int):
    """Retrieve top-k, excluding: exact-answer-match, same-patient, and
    near-duplicate-answer (jaccard >= 0.8) candidates."""
    scores = bm25.get_scores(query_tokens)
    ranked = scores.argsort()[::-1]
    self_answer_norm = self_answer.strip()
    self_answer_tokens = tokenize_vi(self_answer)

    results = []
    for idx in ranked:
        cand = metadata[idx]
        cand_answer = cand["answer"].strip()
        if cand_answer == self_answer_norm:
            continue  # exact-match self
        if self_ma_ho_so and cand.get("ma_ho_so", "") == self_ma_ho_so:
            continue  # same patient
        if jaccard(tokenize_vi(cand_answer), self_answer_tokens) >= 0.8:
            continue  # near-duplicate answer
        results.append(cand)
        if len(results) == top_k:
            break
    return results


def build_augmented_question(original_question: str, retrieved: list) -> str:
    cases = "\n".join([f"- {c['answer']}" for c in retrieved])
    return f"{original_question}\n\nChẩn đoán tham khảo từ ca tương tự:\n{cases}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  default="data/dataset_train.jsonl")
    parser.add_argument("--index_dir",  default="rag_index_strict/")
    parser.add_argument("--output",     default="data/dataset_train_rag_strict.jsonl")
    parser.add_argument("--top_k",      type=int, default=3)
    parser.add_argument("--aug_ratio",  type=float, default=1.0)
    args = parser.parse_args()

    samples = []
    with open(args.data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} training samples")

    with open(os.path.join(args.index_dir, "q_bm25.pkl"), "rb") as f:
        q_bm25 = pickle.load(f)

    metadata = []
    with open(os.path.join(args.index_dir, "metadata.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))
    print(f"BM25 index: {len(metadata)} docs")

    import random
    random.seed(42)

    augmented = 0
    kept_original = 0
    n_fewer_than_k = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for s in samples:
            if random.random() > args.aug_ratio:
                out.write(json.dumps(s, ensure_ascii=False) + "\n")
                kept_original += 1
                continue

            query_tokens = tokenize_vi(s["question"])
            retrieved = retrieve_excluding_self_strict(
                q_bm25, metadata, query_tokens,
                self_answer=s["answer"], self_ma_ho_so=s.get("ma_ho_so", ""),
                top_k=args.top_k)

            if not retrieved:
                out.write(json.dumps(s, ensure_ascii=False) + "\n")
                kept_original += 1
                continue
            if len(retrieved) < args.top_k:
                n_fewer_than_k += 1

            record = dict(s)
            record["question"] = build_augmented_question(s["question"], retrieved)
            record["original_question"] = s["question"]
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            augmented += 1

    total = augmented + kept_original
    print(f"\nDone -> {args.output}")
    print(f"  Augmented (with RAG):  {augmented}/{total} ({100*augmented/total:.1f}%)")
    print(f"  Original (no RAG):     {kept_original}/{total} ({100*kept_original/total:.1f}%)")
    print(f"  Augmented w/ <{args.top_k} candidates (stricter filter exhausted pool): {n_fewer_than_k}")


if __name__ == "__main__":
    main()
