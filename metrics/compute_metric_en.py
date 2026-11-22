"""
Proxy classification metrics — ENGLISH version of proxy_classification_metrics.py.

Same methodology (rule-based keyword matching -> 5 disease groups + laterality
accuracy), reimplemented with an English keyword table so it can be applied to
the English-translated pipeline (baseline/LoRA-FT/XAttn trained+evaluated on
data_enhanced_en/). Kept as a SEPARATE file (not a flag on the Vietnamese
script) because the keyword tables are language-specific and mixing them would
silently corrupt matching for both languages.

Usage:
  python compute_metric_en.py \
      --files results_en/baseline_en_outputs_FIXED.jsonl \
              results_en/lora_ft_en_outputs_FIXED.jsonl \
              results_en/xattn_minimal_en_outputs.jsonl \
      --names Baseline-EN LoRA-FT-EN XAttn-EN \
      --output_json results_en/proxy_classification_metrics_en.json \
      --output_md   results_en/proxy_classification_analysis_en.md
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Keyword table — English translation of the Vietnamese table in
# proxy_classification_metrics.py, same 5 groups / same weights, so results
# are directly comparable in methodology (not in absolute magnitude, since
# English clinical phrasing differs from the Vietnamese source).
# ---------------------------------------------------------------------------
KEYWORDS: Dict[str, List[Tuple[str, int]]] = {
    "U_XUONG": [  # bone tumor
        (r"sarcoma", 10), (r"osteosarcoma", 10), (r"chondrosarcoma", 10),
        (r"bone cancer", 10), (r"bone metastas[ei]s", 10), (r"metastatic", 8),
        (r"cancer", 8), (r"malignan\w*", 8), (r"lymphoma", 8), (r"myeloma", 8),
        (r"leukemi\w*", 8),
        (r"bone tumor", 9), (r"bone tumour", 9), (r"cartilage tumor", 9),
        (r"benign tumor", 7), (r"benign tumour", 7), (r"malignant tumor", 9),
        (r"bone cyst", 8), (r"fibrous dysplasia", 8), (r"dysplasia", 6),
        (r"giant cell tumor", 9), (r"giant cell tumour", 9),
        (r"hemangioma", 7), (r"lipoma", 7), (r"neuroma", 7),
        (r"chondroma", 9), (r"osteoma", 9), (r"enchondroma", 9),
        (r"exostosis", 8), (r"osteochondroma", 8),
        (r"pathologic\w* fracture.{0,40}(tumor|tumour|dysplasia|sarcoma)", 9),
    ],
    "NHIEM_TRUNG": [  # infection
        (r"bone infection", 10), (r"osteomyelitis", 10),
        (r"septic arthritis", 9), (r"joint infection", 9),
        (r"bone abscess", 9), (r"abscess", 6), (r"pus", 6),
        (r"surgical site infection", 7), (r"infected", 7),
        (r"bone necrosis", 7), (r"avascular necrosis", 6),
        (r"tuberculosis", 9), (r"tb (bone|joint|spine)", 9),
        (r"brucell\w*", 9), (r"antibiotic cement", 5), (r"culture positive", 4),
        (r"cellulitis", 6), (r"purulent", 6),
    ],
    "THOAI_HOA": [  # degenerative
        (r"degenerat\w*", 10), (r"scoliosis", 9), (r"spondylolisthesis", 9),
        (r"osteophyte", 8), (r"herniat\w* disc", 9), (r"disc herniation", 9),
        (r"spinal stenosis", 8), (r"rheumatoid arthritis", 9),
        (r"arthritis", 6), (r"osteoarthritis", 8),
        (r"carpal tunnel syndrome", 7), (r"tunnel syndrome", 6), (r"gout", 7),
        (r"calcification", 5), (r"sclerosis", 5), (r"syringomyelia", 7),
        (r"kyphosis", 8), (r"spondylosis", 9),
        (r"subluxation", 7), (r"disc disease", 8), (r"disc prolapse", 8),
        (r"sciatica", 6), (r"nerve root compression", 6),
    ],
    "CHUYEN_HOA": [  # metabolic
        (r"osteoporosis", 10), (r"osteopenia", 8), (r"rickets", 10),
        (r"osteomalacia", 10), (r"hyperparathyroidism", 10),
        (r"hypoparathyroidism", 10),
        (r"paget'?s? disease", 9), (r"metabolic bone disease", 10),
        (r"bone metabolism disorder", 10), (r"vitamin d deficiency", 8),
        (r"hypocalcemia", 7), (r"hypercalcemia", 6), (r"fluorosis", 9),
        (r"renal osteodystrophy", 9),
        (r"pathologic\w* fracture.{0,40}osteoporosis", 9),
    ],
    "CHAN_THUONG": [  # trauma
        (r"fracture", 4), (r"dislocation", 5), (r"rupture", 4),
        (r"compression fracture", 5), (r"tendon rupture", 6),
        (r"tendon tear", 6), (r"ligament rupture", 6), (r"ligament tear", 6),
        (r"trauma", 4), (r"accident", 3), (r"fall(en)?", 3),
        (r"sprain", 5), (r"hematoma", 4), (r"haematoma", 4),
        (r"internal fixation", 3), (r"plate and screw", 3), (r"screws?", 2),
        (r"intramedullary nail", 4), (r"wound", 3), (r"exposed bone", 4),
        (r"casted?", 3), (r"cast(ing)?", 2),
    ],
}

GROUPS = list(KEYWORDS.keys())


def score_text(text: str) -> Dict[str, int]:
    text_lower = text.lower()
    scores = {g: 0 for g in GROUPS}
    for group, patterns in KEYWORDS.items():
        for pattern, weight in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                scores[group] += weight * min(len(matches), 3)
    return scores


def assign_label(text: str) -> Optional[str]:
    scores = score_text(text)
    max_score = max(scores.values())
    if max_score == 0:
        return None
    top = [g for g, s in scores.items() if s == max_score]
    return top[0]


LATERALITY_RE = {
    "left": r"\bleft\b",
    "right": r"\bright\b",
}


def laterality_side(text: str) -> Optional[str]:
    text_lower = text.lower()
    has_left = bool(re.search(LATERALITY_RE["left"], text_lower))
    has_right = bool(re.search(LATERALITY_RE["right"], text_lower))
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    return None


# ---------------------------------------------------------------------------
# Metrics (identical logic to proxy_classification_metrics.py)
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def compute_proxy_metrics(records: List[dict]) -> dict:
    labels_all = GROUPS + ["UNCLASSIFIED"]
    confusion = {r: {p: 0 for p in labels_all} for r in labels_all}

    n_total = len(records)
    n_ref_classified = 0
    n_exact_match = 0

    lat_total = 0
    lat_correct = 0
    lat_opposite_error = 0

    for rec in records:
        ref_text = rec.get("reference", "") or ""
        pred_text = rec.get("prediction", "") or ""

        ref_label = assign_label(ref_text) or "UNCLASSIFIED"
        pred_label = assign_label(pred_text) or "UNCLASSIFIED"

        if ref_label != "UNCLASSIFIED":
            n_ref_classified += 1
            confusion[ref_label][pred_label] += 1
            if ref_label == pred_label:
                n_exact_match += 1

        ref_side = laterality_side(ref_text)
        if ref_side is not None:
            lat_total += 1
            pred_side = laterality_side(pred_text)
            if pred_side == ref_side:
                lat_correct += 1
            elif pred_side is not None and pred_side != ref_side:
                lat_opposite_error += 1

    per_class = {}
    for g in GROUPS:
        tp = confusion[g][g]
        fn = sum(confusion[g][p] for p in labels_all) - tp
        fp = sum(confusion[r][g] for r in labels_all) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[g] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "n_ref": sum(confusion[g][p] for p in labels_all),
        }

    macro_f1 = sum(per_class[g]["f1"] for g in GROUPS) / len(GROUPS)
    accuracy = n_exact_match / n_ref_classified if n_ref_classified else 0.0
    laterality_accuracy = lat_correct / lat_total if lat_total else None
    laterality_opposite_rate = lat_opposite_error / lat_total if lat_total else None

    return {
        "n_total_samples": n_total,
        "n_ref_classified": n_ref_classified,
        "n_ref_unclassified": n_total - n_ref_classified,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "laterality": {
            "n_samples_with_clear_side": lat_total,
            "accuracy": round(laterality_accuracy, 4) if laterality_accuracy is not None else None,
            "opposite_side_error_rate": round(laterality_opposite_rate, 4) if laterality_opposite_rate is not None else None,
        },
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def render_confusion_table(confusion: dict) -> str:
    labels = GROUPS + ["UNCLASSIFIED"]
    header = "| Ref \\ Pred | " + " | ".join(labels) + " |"
    sep = "|---" * (len(labels) + 1) + "|"
    rows = [header, sep]
    for r in labels:
        row = f"| **{r}** | " + " | ".join(str(confusion[r][p]) for p in labels) + " |"
        rows.append(row)
    return "\n".join(rows)


def render_markdown(all_results: Dict[str, dict], output_md: str):
    lines = []
    lines.append("# Proxy Classification Metrics (English pipeline)")
    lines.append("")
    lines.append(
        "English counterpart of `proxy_classification_metrics.py` — same rule-based "
        "keyword-matching methodology (5 disease groups + laterality accuracy), "
        "reimplemented with an English keyword table for the English-translated "
        "baseline/LoRA-FT/XAttn pipeline. Complements BERTScore, which only measures "
        "semantic similarity, not clinical correctness."
    )
    lines.append("")

    summary_rows = []
    for name, res in all_results.items():
        lat = res["laterality"]
        summary_rows.append(
            f"| {name} | {res['n_ref_classified']}/{res['n_total_samples']} "
            f"| {res['accuracy']:.4f} | {res['macro_f1']:.4f} "
            f"| {lat['accuracy'] if lat['accuracy'] is not None else 'N/A'} "
            f"| {lat['opposite_side_error_rate'] if lat['opposite_side_error_rate'] is not None else 'N/A'} |"
        )

    lines.append("## Summary")
    lines.append("")
    lines.append("| Model | N (labeled/total) | Accuracy (5 groups) | Macro-F1 | Laterality Acc | Wrong-side rate |")
    lines.append("|---|---|---|---|---|---|")
    lines.extend(summary_rows)
    lines.append("")

    for name, res in all_results.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- Total samples: {res['n_total_samples']}, labeled from reference: {res['n_ref_classified']} "
                      f"({res['n_ref_unclassified']} references matched no keyword — excluded from accuracy/F1).")
        lines.append(f"- **Accuracy (5 disease groups): {res['accuracy']:.4f}**")
        lines.append(f"- **Macro-F1: {res['macro_f1']:.4f}**")
        lat = res["laterality"]
        if lat["accuracy"] is not None:
            lines.append(f"- **Laterality accuracy: {lat['accuracy']:.4f}** (on {lat['n_samples_with_clear_side']} samples with a clear side in the reference)")
            lines.append(f"- **Wrong-side rate (most dangerous error): {lat['opposite_side_error_rate']:.4f}**")
        lines.append("")
        lines.append("**Per-class Precision / Recall / F1:**")
        lines.append("")
        lines.append("| Group | N (ref) | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|---|")
        for g in GROUPS:
            pc = res["per_class"][g]
            lines.append(f"| {g} | {pc['n_ref']} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} |")
        lines.append("")
        lines.append("**Confusion matrix (rows = reference, cols = prediction):**")
        lines.append("")
        lines.append(render_confusion_table(res["confusion_matrix"]))
        lines.append("")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown report saved -> {output_md}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output_json", default="results_en/proxy_classification_metrics_en.json")
    parser.add_argument("--output_md", default="results_en/proxy_classification_analysis_en.md")
    args = parser.parse_args()

    assert len(args.files) == len(args.names)

    all_results = {}
    for name, path in zip(args.names, args.files):
        print(f"Processing {name}: {path}")
        records = load_jsonl(path)
        all_results[name] = compute_proxy_metrics(records)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"JSON saved -> {args.output_json}")

    render_markdown(all_results, args.output_md)


if __name__ == "__main__":
    main()
