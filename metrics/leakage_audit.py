"""
Leakage audit (THESIS_REVISION_PLAN.md, muc moi 2026-06-18).

Kiem tra xem cau tra loi chuan (reference) co bi "lo" nguyen van (hoac gan
nguyen van) ngay trong phan ngu canh lam sang cua cau hoi (question)
khong. Neu co, model co the dat diem cao chi bang cach copy text, khong
can suy luan tu anh/ngu canh thuc su.

Dong thoi: tinh lai proxy classification accuracy/macro-F1 TACH RIENG cho
tap "leak" va tap "khong leak", de kiem tra khoang cach XAttn vs Qwen-VL
(minimal-context) co con giu duoc tren tap sach (khong leak) hay khong.

Usage:
  python leakage_audit.py \
      --ref_file   results/outputs/xattn_minimal_trained_outputs.jsonl \
      --compare_files results/outputs/xattn_minimal_trained_outputs.jsonl results/outputs/qwenvl_minimal_outputs.jsonl \
      --compare_names XAttn-Minimal-Trained QwenVL-Zeroshot-Minimal \
      --output_md results/leakage_audit_minimal.md

--ref_file: file dung de lay field "question" + "reference" va tinh leak flag
            (line order phai khop voi --compare_files, vi deu xuat phat tu
            cung 1 dataset_test*.jsonl)
"""

import argparse
import json
import os
import re
import unicodedata
from typing import List, Optional

# Tai su dung logic phan loai tu proxy_classification_metrics.py
from proxy_classification_metrics import assign_label, laterality_side, GROUPS


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.lower()
    text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_containment_ratio(ref_norm: str, question_norm: str) -> float:
    """Ti le token cua reference cung xuat hien trong question (theo set, khong thu tu)."""
    ref_tokens = ref_norm.split()
    if not ref_tokens:
        return 0.0
    q_tokens = set(question_norm.split())
    matched = sum(1 for t in ref_tokens if t in q_tokens)
    return matched / len(ref_tokens)


def is_leaked(question: str, reference: str, containment_threshold: float = 0.85) -> dict:
    ref_norm = normalize(reference)
    q_norm = normalize(question)

    if len(ref_norm) < 4:  # reference qua ngan (vd "Co", "Khong") -> khong xet
        return {"leak": False, "reason": "ref_too_short", "containment": 0.0, "exact_substring": False}

    exact = ref_norm in q_norm
    containment = token_containment_ratio(ref_norm, q_norm)

    leak = exact or containment >= containment_threshold
    reason = "exact_substring" if exact else ("high_token_containment" if leak else "no_leak")
    return {"leak": leak, "reason": reason, "containment": round(containment, 3), "exact_substring": exact}


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------------------
# Subset proxy classification (tai su dung tu proxy_classification_metrics)
# ---------------------------------------------------------------------------

def compute_proxy_on_subset(records: List[dict], indices: List[int]) -> dict:
    n = len(indices)
    n_ref_classified = 0
    n_correct = 0
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
            if ref_label == (pred_label or ""):
                n_correct += 1
        ref_side = laterality_side(ref_text)
        if ref_side is not None:
            lat_total += 1
            if laterality_side(pred_text) == ref_side:
                lat_correct += 1

    return {
        "n_samples": n,
        "n_ref_classified": n_ref_classified,
        "accuracy": round(n_correct / n_ref_classified, 4) if n_ref_classified else None,
        "laterality_n": lat_total,
        "laterality_accuracy": round(lat_correct / lat_total, 4) if lat_total else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_file", required=True, help="File de lay question+reference va tinh leak flag")
    parser.add_argument("--compare_files", nargs="+", required=True)
    parser.add_argument("--compare_names", nargs="+", required=True)
    parser.add_argument("--containment_threshold", type=float, default=0.85)
    parser.add_argument("--output_md", default="results/leakage_audit.md")
    parser.add_argument("--output_json", default="results/leakage_audit.json")
    args = parser.parse_args()

    assert len(args.compare_files) == len(args.compare_names)

    ref_records = load_jsonl(args.ref_file)
    n = len(ref_records)
    print(f"Loaded {n} samples from --ref_file: {args.ref_file}")

    leak_flags = []
    leak_examples = []
    for i, rec in enumerate(ref_records):
        result = is_leaked(rec.get("question", ""), rec.get("reference", ""), args.containment_threshold)
        leak_flags.append(result["leak"])
        if result["leak"] and len(leak_examples) < 8:
            leak_examples.append({
                "index": i,
                "image": rec.get("image", ""),
                "question": rec.get("question", "")[:300],
                "reference": rec.get("reference", "")[:200],
                "reason": result["reason"],
                "containment": result["containment"],
            })

    n_leak = sum(leak_flags)
    leak_rate = n_leak / n if n else 0.0
    leak_idx = [i for i, f in enumerate(leak_flags) if f]
    clean_idx = [i for i, f in enumerate(leak_flags) if not f]

    print(f"Leak rate: {n_leak}/{n} = {leak_rate:.4f}")

    comparison = {}
    for name, path in zip(args.compare_names, args.compare_files):
        records = load_jsonl(path)
        assert len(records) == n, f"{path} co {len(records)} dong, khac voi ref_file ({n} dong) -> khong the gan leak flag theo line index"
        comparison[name] = {
            "overall": compute_proxy_on_subset(records, list(range(n))),
            "leak_subset": compute_proxy_on_subset(records, leak_idx),
            "clean_subset": compute_proxy_on_subset(records, clean_idx),
        }

    output = {
        "ref_file": args.ref_file,
        "n_total": n,
        "n_leak": n_leak,
        "leak_rate": round(leak_rate, 4),
        "leak_examples": leak_examples,
        "comparison": comparison,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"JSON saved -> {args.output_json}")

    # ---- Markdown ----
    lines = []
    lines.append("# Leakage Audit")
    lines.append("")
    lines.append(f"File dung de tinh leak flag (`question`+`reference`): `{args.ref_file}`")
    lines.append("")
    lines.append(f"**Ty le ro ri (leak rate): {n_leak}/{n} = {leak_rate:.2%}**")
    lines.append("")
    lines.append("Leak = reference xuat hien nguyen van (substring) trong question, "
                  f"HOAC >= {args.containment_threshold:.0%} token cua reference cung xuat hien trong question.")
    lines.append("")

    lines.append("## So sanh accuracy: Toan bo vs Leak-subset vs Clean-subset")
    lines.append("")
    lines.append("| Model | Accuracy (toàn bộ) | Accuracy (leak, n=" + str(len(leak_idx)) + ") | **Accuracy (clean/sạch, n=" + str(len(clean_idx)) + ")** | Laterality (toàn bộ) | Laterality (clean) |")
    lines.append("|---|---|---|---|---|---|")
    for name, res in comparison.items():
        o, lk, cl = res["overall"], res["leak_subset"], res["clean_subset"]
        lines.append(
            f"| {name} | {o['accuracy']} | {lk['accuracy']} | **{cl['accuracy']}** "
            f"| {o['laterality_accuracy']} | {cl['laterality_accuracy']} |"
        )
    lines.append("")

    lines.append(f"## Vi du mau bi leak (toi da 8/{n_leak})")
    lines.append("")
    for ex in leak_examples:
        lines.append(f"**Sample index {ex['index']}** (lý do: {ex['reason']}, containment={ex['containment']})")
        lines.append(f"- image: `{ex['image']}`")
        lines.append(f"- question: {ex['question']}")
        lines.append(f"- reference: {ex['reference']}")
        lines.append("")

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown saved -> {args.output_md}")


if __name__ == "__main__":
    main()
