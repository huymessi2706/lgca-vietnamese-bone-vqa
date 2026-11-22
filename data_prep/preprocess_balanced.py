"""
Tạo dataset cân bằng từ data_enhanced đã có sẵn.

Vấn đề mà script này giải quyết:
  - Dataset X-quang xương bị dominated bởi gãy xương và thoái hóa cột sống
  - Model học lệch → overpredict common classes
  - Cross-attention modules (frozen LLM) dễ overfit trên rare classes nếu underrepresented

Chiến lược cân bằng:
  1. Phân loại mỗi sample vào disease category bằng keyword matching trên answer
  2. Cap số lượng tối đa mỗi class (--max_per_class)
  3. Giữ toàn bộ nếu class có ít hơn min_keep_threshold samples
  4. Shuffle ngẫu nhiên (--seed) để reproducible

Ideal data size cho XAttn fine-tuning (17 modules, frozen LLM):
  - Sweet spot: 10,000–20,000 samples với class balance tốt
  - Max per class: 800–1000 để tránh domination
  - Min per class: giữ tất cả nếu < 200 (rare diseases vẫn cần coverage)

Usage:
    python preprocess_balanced.py \\
        --input      data_enhanced/dataset_train.jsonl \\
        --output     data_enhanced/dataset_train_balanced.jsonl \\
        --max_per_class 800 \\
        --max_total  15000 \\
        --seed       42

    # Sau đó generate minimal version:
    python build_minimal_test.py \\
        --input  data_enhanced/dataset_train_balanced.jsonl \\
        --output data_enhanced/dataset_train_balanced_minimal.jsonl
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# Disease category definitions (keyword matching trên answer text)
# Thứ tự quan trọng: categories đặt trước có priority cao hơn
# ---------------------------------------------------------------------------

CATEGORIES = [
    ("post_surgery",    ["đã mổ", "kết hợp xương", "sau mổ", "nẹp vít",
                         "đinh nội tủy", "đinh kirschner", "bó bột", "tháo nẹp",
                         "bất động", "xuyên kim"]),
    ("fracture",        ["gãy", "vỡ xương", "nứt xương", "gãy hở", "gãy kín",
                         "gãy lún", "gãy nát", "gãy cành tươi"]),
    ("degeneration",    ["thoái hóa", "hẹp khe khớp", "mất chiều cao đĩa đệm",
                         "thoát vị đĩa đệm", "gai xương", "xẹp đĩa đệm",
                         "hẹp ống sống"]),
    ("inflammation",    ["viêm khớp", "viêm cột sống", "viêm thoái hóa",
                         "viêm quanh khớp", "dính khớp", "viêm gân",
                         "viêm cân gan bàn chân"]),
    ("spinal_nerve",    ["thần kinh tọa", "đau thần kinh", "chèn ép rễ",
                         "hội chứng thần kinh", "liệt", "đau dây thần kinh",
                         "tê bì", "yếu chi"]),
    ("dislocation",     ["trật khớp", "sai khớp", "bán trật", "trật",
                         "lệch khớp"]),
    ("necrosis",        ["hoại tử", "thiếu máu chỏm", "nhồi máu xương"]),
    ("tumor",           ["bướu", "u xương", "u sụn", "ung thư xương",
                         "di căn", "khối u", "nang xương", "bướu sụn"]),
    ("infection",       ["nhiễm trùng", "áp xe xương", "viêm tủy xương",
                         "osteomyelitis", "nhiễm khuẩn"]),
    ("metabolic",       ["loãng xương", "thiếu canxi", "còi xương",
                         "bệnh paget", "cường giáp"]),
    ("treatment_plan",  ["cho toa thuốc", "toa thuốc về", "hướng điều trị",
                         "phương pháp điều trị"]),
    ("other",           []),  # catch-all
]


def classify_answer(answer: str) -> str:
    """Phân loại answer vào disease category đầu tiên khớp."""
    a = answer.lower()
    for category, keywords in CATEGORIES:
        if category == "other":
            return "other"
        for kw in keywords:
            if kw in a:
                return category
    return "other"


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def print_distribution(counts: Counter, title: str, total: int):
    print(f"\n{'='*55}")
    print(f"{title} (total={total})")
    print(f"{'─'*55}")
    print(f"  {'Category':<20} {'Count':>7}  {'%':>6}")
    print(f"  {'─'*20} {'─'*7}  {'─'*6}")
    for cat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / total if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {cat:<20} {cnt:>7}  {pct:>5.1f}%  {bar}")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Balance disease class distribution in training data"
    )
    parser.add_argument("--input",   default="data_enhanced/dataset_train.jsonl",
                        help="Input JSONL (from preprocess_enhanced.py)")
    parser.add_argument("--output",  default="data_enhanced/dataset_train_balanced.jsonl",
                        help="Output balanced JSONL")
    parser.add_argument("--max_per_class", type=int, default=800,
                        help="Max samples per disease class (default 800). "
                             "Classes với ít hơn ngưỡng này không bị cắt.")
    parser.add_argument("--max_total", type=int, default=None,
                        help="Hard cap tổng số samples. Nếu sau per-class cap "
                             "vẫn nhiều hơn max_total, sample thêm đều giữa các class.")
    parser.add_argument("--min_keep", type=int, default=100,
                        help="Nếu class có ít hơn min_keep samples, giữ tất cả "
                             "dù có vượt max_per_class (rare class protection). "
                             "Default 100.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Load ─────────────────────────────────────────────────────────────────
    samples = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples from {args.input}")

    # ── Classify ─────────────────────────────────────────────────────────────
    buckets: dict[str, list] = defaultdict(list)
    for s in samples:
        cat = classify_answer(s.get("answer", ""))
        s["_category"] = cat  # temp field for inspection
        buckets[cat].append(s)

    before_counts = Counter({k: len(v) for k, v in buckets.items()})
    print_distribution(before_counts, "BEFORE balancing", len(samples))

    # ── Cap per class ─────────────────────────────────────────────────────────
    balanced = []
    after_counts = Counter()

    for cat, items in sorted(buckets.items()):
        random.shuffle(items)
        n = len(items)

        if n <= args.min_keep:
            # Rare class: giữ tất cả
            kept = items
            reason = f"rare (≤{args.min_keep})"
        elif n <= args.max_per_class:
            # Đủ nhỏ: giữ tất cả
            kept = items
            reason = "within cap"
        else:
            # Over cap: cắt xuống max_per_class
            kept = items[:args.max_per_class]
            reason = f"capped {n}→{args.max_per_class}"

        after_counts[cat] = len(kept)
        balanced.extend(kept)
        print(f"  [{cat:<20}] {n:>5} → {len(kept):>5}  ({reason})")

    # ── Hard total cap ────────────────────────────────────────────────────────
    if args.max_total and len(balanced) > args.max_total:
        print(f"\nApplying hard max_total cap: {len(balanced)} → {args.max_total}")
        # Stratified reduction: giảm đều các class
        ratio = args.max_total / len(balanced)
        final = []
        for cat, items in sorted(
            {c: [s for s in balanced if s["_category"] == c] for c in after_counts}.items()
        ):
            n_keep = max(1, int(len(items) * ratio))
            final.extend(items[:n_keep])
        # Top up nếu còn thiếu vài samples
        remaining = [s for s in balanced if s not in set(id(x) for x in final)]
        random.shuffle(remaining)
        final.extend(remaining[:args.max_total - len(final)])
        balanced = final[:args.max_total]
        after_counts = Counter(s["_category"] for s in balanced)

    # ── Remove temp field + shuffle final output ──────────────────────────────
    random.shuffle(balanced)
    for s in balanced:
        s.pop("_category", None)

    print_distribution(after_counts, "AFTER balancing", len(balanced))

    # ── Imbalance ratio check ─────────────────────────────────────────────────
    counts_vals = list(after_counts.values())
    if counts_vals:
        ratio = max(counts_vals) / max(min(counts_vals), 1)
        print(f"\nImbalance ratio (max/min): {ratio:.1f}x  ", end="")
        if ratio <= 5:
            print("✓ Good")
        elif ratio <= 10:
            print("⚠ Moderate — consider lowering --max_per_class")
        else:
            print("✗ High — strongly recommend lowering --max_per_class")

    # ── Save ──────────────────────────────────────────────────────────────────
    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in balanced:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(balanced)} samples → {args.output}")
    print(f"\nNext step — generate minimal context version:")
    print(f"  python build_minimal_test.py \\")
    print(f"      --input  {args.output} \\")
    out_minimal = args.output.replace(".jsonl", "_minimal.jsonl")
    print(f"      --output {out_minimal}")


if __name__ == "__main__":
    main()
