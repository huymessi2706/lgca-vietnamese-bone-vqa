"""
Proxy classification metrics (THESIS_REVISION_PLAN.md, muc #5 + #3).

Khong train classifier moi. Map ca prediction va reference ve 1 trong 5
nhom benh bang dung bo keyword da dung de gan nhan dataset goc
(classify_patients.py), roi tinh accuracy / macro-F1 / confusion matrix /
laterality accuracy giua 2 nhan do.

Day la thuoc do "dung-sai lam sang" o muc tho (proxy), bo sung cho
BERTScore (chi do tuong dong ngu nghia, khong do dung benh).

Usage:
  python proxy_classification_metrics.py \
      --files results/outputs/qwenvl_full_outputs.jsonl results/outputs/qwenvl_minimal_outputs.jsonl \
      --names QwenVL-Full QwenVL-Minimal \
      --output_json results/proxy_classification_metrics.json \
      --output_md   results/proxy_classification_analysis.md
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Keyword table - sao chep tu classify_patients.py de nhat quan phuong phap
# (chi doi ten nhom cho khop voi 5 nhom dung trong chapter3: CHAN_THUONG,
# THOAI_HOA, NHIEM_TRUNG, U_XUONG, CHUYEN_HOA - bo suffix _XUONG)
# ---------------------------------------------------------------------------
KEYWORDS: Dict[str, List[Tuple[str, int]]] = {
    "U_XUONG": [
        (r"sarcom", 10), (r"sarcoma", 10), (r"osteosarcoma", 10),
        (r"ung thư xương", 10), (r"di căn xương", 10), (r"ung thư", 8),
        (r"ác tính", 8), (r"lymphoma", 8), (r"myeloma", 8), (r"leuk[eê]mi", 8),
        (r"u xương", 9), (r"u sụn", 9), (r"u lành", 7), (r"bướu xương", 9),
        (r"bướu lành", 7), (r"bướu ác", 9), (r"bọc xương", 8), (r"nang xương", 8),
        (r"loạn sản sợi", 8), (r"loạn sản", 6), (r"u tế bào khổng lồ", 9),
        (r"giant cell", 9), (r"u mạch", 7), (r"u mỡ", 7), (r"u thần kinh", 7),
        (r"chondroma", 9), (r"osteoma", 9), (r"enchondroma", 9),
        (r"exostosis", 8), (r"osteochondroma", 8),
        (r"gãy bệnh lý.{0,40}(u|bướu|loạn sản|sarcom)", 9),
    ],
    "NHIEM_TRUNG": [
        (r"nhiễm trùng xương", 10), (r"viêm xương", 10), (r"viêm tủy xương", 10),
        (r"osteomyelitis", 10), (r"viêm khớp nhiễm", 9), (r"nhiễm trùng khớp", 9),
        (r"áp xe xương", 9), (r"áp xe", 6), (r"mủ xương", 9), (r"mủ khớp", 9),
        (r"nhiễm trùng vết mổ", 7), (r"nhiễm khuẩn xương", 9),
        (r"hoại tử xương", 7), (r"septic arthritis", 9), (r"lao xương", 9),
        (r"lao khớp", 9), (r"brucella", 9), (r"xi măng kháng sinh", 5),
        (r"cấy m[ủu]", 4),
    ],
    "THOAI_HOA": [
        (r"thoái hóa", 10), (r"thoái hoá", 10), (r"vẹo cột sống", 9),
        (r"trượt đốt sống", 9), (r"gai cột sống", 8), (r"thoát vị đĩa đệm", 9),
        (r"hẹp ống sống", 8), (r"viêm khớp dạng thấp", 9), (r"viêm khớp", 6),
        (r"hội chứng ống cổ tay", 7), (r"hội chứng đường hầm", 7), (r"gout", 7),
        (r"gút", 7), (r"calci hóa", 5), (r"xơ cứng", 5), (r"rỗng tủy", 7),
        (r"scoliosis", 8), (r"kyphosis", 8), (r"gù cột sống", 8), (r"vẹo", 4),
        (r"trật đốt sống", 7), (r"bệnh đĩa đệm", 8), (r"thoát vị nhân nhầy", 8),
        (r"đau thần kinh tọa", 6), (r"đau thần kinh toạ", 6),
    ],
    "CHUYEN_HOA": [
        (r"loãng xương", 10), (r"loang xương", 10), (r"còi xương", 10),
        (r"nhuyễn xương", 10), (r"osteoporosis", 10), (r"osteomalacia", 10),
        (r"rickets", 10), (r"cường cận giáp", 10), (r"suy cận giáp", 10),
        (r"paget", 9), (r"bệnh xương chuyển hóa", 10),
        (r"rối loạn chuyển hóa xương", 10), (r"thiếu vitamin d", 8),
        (r"hạ calci", 7), (r"tăng calci", 6), (r"fluorosis", 9),
        (r"bệnh xương do thận", 9), (r"gãy bệnh lý.{0,40}loãng xương", 9),
    ],
    "CHAN_THUONG": [
        (r"gãy", 4), (r"gay", 3), (r"trật khớp", 5), (r"vỡ", 4), (r"lún", 5),
        (r"đứt gân", 6), (r"rách gân", 6), (r"đứt dây chằng", 6),
        (r"rách dây chằng", 6), (r"chấn thương", 4), (r"tai nạn", 3),
        (r"té ngã", 3), (r"bong gân", 5), (r"hematoma", 4), (r"tụ máu", 4),
        (r"kết hợp xương", 3), (r"nẹp vít", 3), (r"đinh nội tủy", 4),
        (r"vết thương", 3), (r"lộ xương", 4),
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
    """Single-label assignment: nhom co score cao nhat, None neu khong khop gi."""
    scores = score_text(text)
    max_score = max(scores.values())
    if max_score == 0:
        return None
    top = [g for g, s in scores.items() if s == max_score]
    return top[0]


LATERALITY_RE = {"trái": r"\btrái\b", "phải": r"\bphải\b"}


def laterality_side(text: str) -> Optional[str]:
    """Tra ve 'trái'/'phải' neu text chi nhac dung 1 ben, None neu ca 2/khong co."""
    text_lower = text.lower()
    has_trai = bool(re.search(LATERALITY_RE["trái"], text_lower))
    has_phai = bool(re.search(LATERALITY_RE["phải"], text_lower))
    if has_trai and not has_phai:
        return "trái"
    if has_phai and not has_trai:
        return "phải"
    return None  # ca 2 ben hoac khong co thong tin ben -> bo qua mau nay


# ---------------------------------------------------------------------------
# Metrics
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
    lat_opposite_error = 0  # du doan sai HAN sang ben doi dien (nguy hiem)

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

    # Per-class precision/recall/F1 (single-label multi-class, dua tren confusion)
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
    lines.append("# Proxy Classification Metrics - Phan tich")
    lines.append("")
    lines.append(
        "Khong train classifier moi. Map prediction/reference ve 1 trong 5 nhom benh "
        "bang keyword (dung lai bo tu khoa cua `classify_patients.py`), roi tinh "
        "accuracy/macro-F1/confusion matrix/laterality accuracy. Day la thuoc do "
        "**dung-sai lam sang o muc tho**, bo sung cho BERTScore (chi do tuong dong "
        "ngu nghia, khong phan biet duoc dung benh hay sai benh)."
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

    lines.append("## Bang tong hop")
    lines.append("")
    lines.append("| Model | N (co nhan/tong) | Accuracy (5 nhom) | Macro-F1 | Laterality Acc | Laterality sai-ben (%) |")
    lines.append("|---|---|---|---|---|---|")
    lines.extend(summary_rows)
    lines.append("")

    for name, res in all_results.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- Tong mau: {res['n_total_samples']}, co nhan xac dinh duoc tu reference: {res['n_ref_classified']} "
                      f"({res['n_ref_unclassified']} reference khong khop tu khoa nao - bi loai khoi accuracy/F1).")
        lines.append(f"- **Accuracy (5 nhom benh): {res['accuracy']:.4f}**")
        lines.append(f"- **Macro-F1: {res['macro_f1']:.4f}**")
        lat = res["laterality"]
        if lat["accuracy"] is not None:
            lines.append(f"- **Laterality accuracy: {lat['accuracy']:.4f}** (tren {lat['n_samples_with_clear_side']} mau reference co neu ro 1 ben)")
            lines.append(f"- **Ty le doan NGUOC ben (nguy hiem nhat): {lat['opposite_side_error_rate']:.4f}**")
        lines.append("")
        lines.append("**Per-class Precision / Recall / F1:**")
        lines.append("")
        lines.append("| Nhom | N (ref) | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|---|")
        for g in GROUPS:
            pc = res["per_class"][g]
            lines.append(f"| {g} | {pc['n_ref']} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} |")
        lines.append("")
        lines.append("**Confusion matrix (hang = reference, cot = prediction):**")
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
    parser.add_argument("--output_json", default="results/proxy_classification_metrics.json")
    parser.add_argument("--output_md", default="results/proxy_classification_analysis.md")
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
