"""
Bootstrap Confidence Interval (THESIS_REVISION_PLAN.md, muc #4a).

Tra loi cau hoi: chenh lech giua 2 model (vd XAttn vs LoRA-QwenVL) tren proxy
accuracy / macro-F1 / laterality accuracy co vuot qua nhieu mau hay khong?

Dung paired bootstrap: moi lan resample, CUNG 1 bo chi so duoc ap dung cho
TAT CA model dang so sanh (dam bao so sanh dung cap, khong phai 2 mau doc
lap) - day la cach lam chuan cho so sanh model tren cung 1 test set.

Khong can GPU - chi load text (prediction/reference) va tinh lai
accuracy/macro-F1/laterality bang keyword (tai su dung tu
proxy_classification_metrics.py).

Usage:
  python bootstrap_ci.py \
      --files results/outputs/xattn_minimal_trained_outputs.jsonl \
              results/outputs/lora_qwenvl_minimal_outputs.jsonl \
      --names XAttn-Minimal LoRA-QwenVL-Minimal \
      --leak_ref_file results/outputs/xattn_minimal_trained_outputs.jsonl \
      --subset clean \
      --n_bootstrap 2000 \
      --output_md results/bootstrap_ci_minimal.md
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional

from proxy_classification_metrics import GROUPS, assign_label, laterality_side
from leakage_audit import is_leaked


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def compute_metrics_for_indices(records: List[dict], indices: List[int]) -> Dict[str, Optional[float]]:
    n_ref_classified = 0
    n_correct = 0
    per_class_tp = {g: 0 for g in GROUPS}
    per_class_fp = {g: 0 for g in GROUPS}
    per_class_fn = {g: 0 for g in GROUPS}
    lat_total = 0
    lat_correct = 0

    for i in indices:
        rec = records[i]
        ref_text = rec.get("reference", "") or ""
        pred_text = rec.get("prediction", "") or ""
        ref_label = assign_label(ref_text)
        pred_label = assign_label(pred_text)

        if ref_label is not None:
            n_ref_classified += 1
            if ref_label == pred_label:
                n_correct += 1
                per_class_tp[ref_label] += 1
            else:
                per_class_fn[ref_label] += 1
                if pred_label in per_class_fp:
                    per_class_fp[pred_label] += 1

        ref_side = laterality_side(ref_text)
        if ref_side is not None:
            lat_total += 1
            if laterality_side(pred_text) == ref_side:
                lat_correct += 1

    accuracy = n_correct / n_ref_classified if n_ref_classified else None
    laterality_accuracy = lat_correct / lat_total if lat_total else None

    f1s = []
    for g in GROUPS:
        tp, fp, fn = per_class_tp[g], per_class_fp[g], per_class_fn[g]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s)

    return {"accuracy": accuracy, "macro_f1": macro_f1, "laterality_accuracy": laterality_accuracy}


def percentile(values: List[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--leak_ref_file", default=None,
                        help="Neu co, dung de tinh leak flag va loc theo --subset")
    parser.add_argument("--subset", choices=["all", "clean", "leak"], default="clean")
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_md", default="results/bootstrap_ci.md")
    parser.add_argument("--output_json", default="results/bootstrap_ci.json")
    args = parser.parse_args()

    assert len(args.files) == len(args.names)
    random.seed(args.seed)

    all_records = {name: load_jsonl(path) for name, path in zip(args.names, args.files)}
    n = len(next(iter(all_records.values())))
    for name, recs in all_records.items():
        assert len(recs) == n, f"{name} co {len(recs)} dong, khac voi {n} -- khong the so sanh cap doi"

    if args.leak_ref_file:
        ref_records = load_jsonl(args.leak_ref_file)
        assert len(ref_records) == n
        leak_flags = [is_leaked(r.get("question", ""), r.get("reference", ""))["leak"] for r in ref_records]
        if args.subset == "clean":
            base_indices = [i for i, f in enumerate(leak_flags) if not f]
        elif args.subset == "leak":
            base_indices = [i for i, f in enumerate(leak_flags) if f]
        else:
            base_indices = list(range(n))
    else:
        base_indices = list(range(n))

    print(f"So mau trong subset '{args.subset}': {len(base_indices)}/{n}")

    # ---- Point estimate (khong resample) ----
    point_estimates = {name: compute_metrics_for_indices(recs, base_indices) for name, recs in all_records.items()}

    # ---- Bootstrap ----
    metric_keys = ["accuracy", "macro_f1", "laterality_accuracy"]
    boot_values = {name: {k: [] for k in metric_keys} for name in args.names}
    boot_diffs = {k: [] for k in metric_keys}  # chi tinh khi co dung 2 model

    n_sub = len(base_indices)
    for b in range(args.n_bootstrap):
        resampled = [random.choice(base_indices) for _ in range(n_sub)]  # cung 1 bo chi so cho moi model
        per_model = {}
        for name, recs in all_records.items():
            m = compute_metrics_for_indices(recs, resampled)
            per_model[name] = m
            for k in metric_keys:
                if m[k] is not None:
                    boot_values[name][k].append(m[k])
        if len(args.names) == 2:
            a, b_name = args.names
            for k in metric_keys:
                va, vb = per_model[a][k], per_model[b_name][k]
                if va is not None and vb is not None:
                    boot_diffs[k].append(va - vb)

    # ---- Tong hop CI ----
    results = {"subset": args.subset, "n_subset": n_sub, "n_bootstrap": args.n_bootstrap,
               "point_estimate": point_estimates, "ci_95": {}, "diff_vs_first": {}}

    for name in args.names:
        results["ci_95"][name] = {}
        for k in metric_keys:
            vals = boot_values[name][k]
            if vals:
                results["ci_95"][name][k] = {
                    "mean": round(sum(vals) / len(vals), 4),
                    "ci_lower": round(percentile(vals, 0.025), 4),
                    "ci_upper": round(percentile(vals, 0.975), 4),
                }

    if len(args.names) == 2:
        a, b_name = args.names
        for k in metric_keys:
            diffs = boot_diffs[k]
            if diffs:
                ci_lo, ci_hi = percentile(diffs, 0.025), percentile(diffs, 0.975)
                # p-value 2-phia xap xi: ti le bootstrap sample co dau nguoc voi diff trung binh
                mean_diff = sum(diffs) / len(diffs)
                if mean_diff >= 0:
                    p_approx = sum(1 for d in diffs if d <= 0) / len(diffs) * 2
                else:
                    p_approx = sum(1 for d in diffs if d >= 0) / len(diffs) * 2
                p_approx = min(p_approx, 1.0)
                results["diff_vs_first"][k] = {
                    "diff_label": f"{a} - {b_name}",
                    "mean_diff": round(mean_diff, 4),
                    "ci_95_lower": round(ci_lo, 4),
                    "ci_95_upper": round(ci_hi, 4),
                    "significant_at_0.05": not (ci_lo <= 0 <= ci_hi),
                    "p_approx_two_sided": round(p_approx, 4),
                }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"JSON saved -> {args.output_json}")

    # ---- Markdown ----
    lines = [
        "# Bootstrap Confidence Interval",
        "",
        f"Subset: **{args.subset}** (n={n_sub}/{n}), n_bootstrap={args.n_bootstrap}, paired (cung bo chi so resample cho moi model).",
        "",
        "## Point estimate + 95% CI (tu bootstrap distribution)",
        "",
        "| Model | Metric | Point estimate | 95% CI |",
        "|---|---|---|---|",
    ]
    for name in args.names:
        for k in metric_keys:
            pe = point_estimates[name][k]
            ci = results["ci_95"].get(name, {}).get(k)
            if pe is not None and ci:
                lines.append(f"| {name} | {k} | {pe:.4f} | [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] |")

    if results["diff_vs_first"]:
        lines.append("")
        lines.append("## Chenh lech giua 2 model (paired bootstrap)")
        lines.append("")
        lines.append("| Metric | Diff (A-B) | 95% CI cua diff | Co y nghia (p<0.05)? | p xap xi |")
        lines.append("|---|---|---|---|---|")
        for k, d in results["diff_vs_first"].items():
            sig = "**CO** (CI khong chua 0)" if d["significant_at_0.05"] else "Khong (CI chua 0 -> co the la nhieu)"
            lines.append(
                f"| {k} | {d['mean_diff']:+.4f} ({d['diff_label']}) | "
                f"[{d['ci_95_lower']:+.4f}, {d['ci_95_upper']:+.4f}] | {sig} | {d['p_approx_two_sided']:.4f} |"
            )

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown saved -> {args.output_md}")


if __name__ == "__main__":
    main()
