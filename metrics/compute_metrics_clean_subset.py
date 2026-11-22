"""
Compute generation metrics (BLEU/ROUGE/METEOR/BERTScore-F1) split by
leak/clean subset, instead of only on the full test set.

Why this script exists: results/leakage_audit_minimal_v2.json only reports
proxy-classification accuracy/macro-F1/laterality split by leak/clean —
it never computed BLEU/ROUGE/METEOR/BERTScore on the clean subset alone.
The thesis's main result table (BERTScore-F1 0.9483/0.9162) is therefore
still measured on the full 1287-sample test set (leak + clean mixed).
chapter4's leakage section currently only has a worst-case ARITHMETIC BOUND
(>=0.891) for what the clean-only BERTScore-F1 could be — this script
replaces that bound with the actual measured number.

Reuses the exact same leak-detection logic as leakage_audit.py (same
containment threshold, same is_leaked() function) so the clean/leak split
here is IDENTICAL to the one already reported for proxy classification —
no risk of the two analyses disagreeing on which samples are "leaked".

Usage:
  python compute_metrics_clean_subset.py \
      --ref_file results/outputs/xattn_minimal_trained_outputs.jsonl \
      --files    results/outputs/xattn_minimal_trained_outputs.jsonl \
                 results/outputs/lora_qwenvl_minimal_outputs.jsonl \
      --names    "XAttn-Minimal-Trained" "LoRA-QwenVL-Minimal" \
      --output_json results/metrics_clean_subset.json \
      --bert_model roberta-large

--ref_file: file used only to compute the leak flag for each line (question +
            reference). Must be the same dataset_test_minimal.jsonl ordering
            as every file in --files (this is true for every minimal-context
            output in this repo — they all come from the same test split).
"""

import argparse
import json
import os

from compute_metrics import compute_all
from leakage_audit import is_leaked, load_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_file", required=True)
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--containment_threshold", type=float, default=0.85)
    p.add_argument("--output_json", default="results/metrics_clean_subset.json")
    p.add_argument("--bert_model", default="roberta-large")
    args = p.parse_args()

    assert len(args.files) == len(args.names)

    ref_records = load_jsonl(args.ref_file)
    n = len(ref_records)
    print(f"Loaded {n} samples from --ref_file: {args.ref_file}")

    leak_flags = [
        is_leaked(rec.get("question", ""), rec.get("reference", ""),
                  args.containment_threshold)["leak"]
        for rec in ref_records
    ]
    leak_idx = [i for i, f in enumerate(leak_flags) if f]
    clean_idx = [i for i, f in enumerate(leak_flags) if not f]
    print(f"Leak rate: {len(leak_idx)}/{n} = {len(leak_idx)/n:.4f}  "
          f"(clean: {len(clean_idx)})")

    results = {}
    for name, path in zip(args.names, args.files):
        records = load_jsonl(path)
        assert len(records) == n, (
            f"{path} has {len(records)} lines, expected {n} to match --ref_file "
            "line order; cannot apply leak flags."
        )
        preds = [r.get("prediction", "") or "" for r in records]
        refs  = [r.get("reference", "")  or "" for r in records]

        print(f"\n=== {name} ===")
        print(f"-- full set (n={n}) --")
        full_metrics = compute_all(preds, refs, args.bert_model)

        leak_preds = [preds[i] for i in leak_idx]
        leak_refs  = [refs[i]  for i in leak_idx]
        print(f"-- leak subset (n={len(leak_idx)}) --")
        leak_metrics = compute_all(leak_preds, leak_refs, args.bert_model)

        clean_preds = [preds[i] for i in clean_idx]
        clean_refs  = [refs[i]  for i in clean_idx]
        print(f"-- clean subset (n={len(clean_idx)}) --")
        clean_metrics = compute_all(clean_preds, clean_refs, args.bert_model)

        results[name] = {
            "n_total": n, "n_leak": len(leak_idx), "n_clean": len(clean_idx),
            "full": full_metrics,
            "leak_subset": leak_metrics,
            "clean_subset": clean_metrics,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {args.output_json}")

    print("\n" + "=" * 70)
    print(f"{'Model':<30} {'Full BERT-F1':>14} {'Leak BERT-F1':>14} {'Clean BERT-F1':>14}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<30} {r['full']['bertscore_F1']:>14.4f} "
              f"{r['leak_subset']['bertscore_F1']:>14.4f} "
              f"{r['clean_subset']['bertscore_F1']:>14.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
