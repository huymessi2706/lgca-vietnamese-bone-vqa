"""
Preprocess + Balance trong một bước duy nhất.

Đọc thẳng từ CSV → generate QA pairs → balance on-the-fly → lưu splits.
Không ghi file dataset trung gian, tiết kiệm disk I/O khi data lớn.

Pipeline:
  CSV + bone_data
    ↓  (preprocess_enhanced logic)
  QA pairs (in memory, grouped by disease class)
    ↓  (per-class cap + max_total)
  Balanced train / val / test JSONL

Usage:
    python preprocess_balanced_direct.py \\
        --image_dir    ./bone_data \\
        --clinical_csv ./Ho_So_Lam_Sang_Day_Du.csv \\
        --output_dir   ./data_enhanced \\
        --max_per_class 800 \\
        --max_total     15000 \\
        --train_ratio   0.75 --val_ratio 0.10 --test_ratio 0.15 \\
        --seed          42

    # Sau đó generate minimal version cho train:
    python build_minimal_test.py \\
        --input  data_enhanced/dataset_train_balanced.jsonl \\
        --output data_enhanced/dataset_train_balanced_minimal.jsonl
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Import parsing logic từ preprocess_enhanced (không thay đổi file đó) ──────
from preprocess_enhanced import (
    COL_MA_HO_SO,
    _clean,
    patient_split,
    build_patient_image_index,
    expand_row_to_visits,
    make_qa_pairs,
    make_diagnosis_answer,
    make_treatment_answer,
    is_valid_image,
)

# ── Import classifier từ preprocess_balanced ──────────────────────────────────
from preprocess_balanced import classify_answer, print_distribution


# ---------------------------------------------------------------------------
# Balance helpers
# ---------------------------------------------------------------------------

def apply_balance(
    buckets: dict,          # {class: [samples]}
    max_per_class: int,
    max_total: int | None,
    min_keep: int,
    seed: int,
) -> list:
    """Cap per-class → optional total cap → shuffle → return flat list."""
    random.seed(seed)
    balanced = []
    after_counts = Counter()

    for cat, items in sorted(buckets.items()):
        random.shuffle(items)
        n = len(items)

        if n <= min_keep:
            kept = items
            reason = f"rare (≤{min_keep})"
        elif n <= max_per_class:
            kept = items
            reason = "within cap"
        else:
            kept = items[:max_per_class]
            reason = f"capped {n}→{max_per_class}"

        after_counts[cat] = len(kept)
        balanced.extend(kept)
        print(f"  [{cat:<20}] {n:>6} → {len(kept):>6}  ({reason})")

    if max_total and len(balanced) > max_total:
        print(f"\nApplying hard max_total: {len(balanced)} → {max_total}")
        ratio = max_total / len(balanced)
        final = []
        # Rebuild per-class for proportional reduction
        per_class = defaultdict(list)
        for s in balanced:
            per_class[s["_category"]].append(s)
        for cat, items in per_class.items():
            n_keep = max(1, int(len(items) * ratio))
            final.extend(items[:n_keep])
        # Fill remaining slots
        used_ids = {id(s) for s in final}
        extras = [s for s in balanced if id(s) not in used_ids]
        random.shuffle(extras)
        final.extend(extras[:max_total - len(final)])
        balanced = final[:max_total]
        after_counts = Counter(s["_category"] for s in balanced)

    # Imbalance check
    vals = list(after_counts.values())
    if vals:
        ratio = max(vals) / max(min(vals), 1)
        label = "✓ Good" if ratio <= 5 else ("⚠ Moderate" if ratio <= 10 else "✗ High")
        print(f"\nImbalance ratio (max/min): {ratio:.1f}x  {label}")

    random.shuffle(balanced)
    # Remove temp classification field
    for s in balanced:
        s.pop("_category", None)

    return balanced


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess + Balance in one step — no intermediate file"
    )
    parser.add_argument("--image_dir",     default="./bone_data")
    parser.add_argument("--clinical_csv",  default="./Ho_So_Lam_Sang_Day_Du.csv")
    parser.add_argument("--output_dir",    default="./data_enhanced")
    parser.add_argument("--train_ratio",   type=float, default=0.75)
    parser.add_argument("--val_ratio",     type=float, default=0.10)
    parser.add_argument("--test_ratio",    type=float, default=0.15)
    parser.add_argument("--max_per_class", type=int,   default=800,
                        help="Max samples per disease class in train split")
    parser.add_argument("--max_total",     type=int,   default=None,
                        help="Hard cap on total train samples after per-class cap")
    parser.add_argument("--min_keep",      type=int,   default=100,
                        help="Keep all samples if class size ≤ this (rare class protection)")
    parser.add_argument("--output_suffix", default="balanced",
                        help="Suffix for output files, e.g. 'balanced' → dataset_train_balanced.jsonl")
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6

    image_dir  = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load CSV ───────────────────────────────────────────────────────────
    print(f"\n[1/4] Đọc CSV: {args.clinical_csv}")
    df = pd.read_csv(args.clinical_csv, dtype=str, encoding="utf-8-sig").fillna("")
    df.columns = [c.strip() for c in df.columns]
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    print(f"  Loaded {len(df)} rows")

    # ── 2. Patient split ──────────────────────────────────────────────────────
    print("\n[2/4] Patient-level split...")
    all_pids = df[COL_MA_HO_SO].dropna().unique().tolist()
    train_pids, val_pids, test_pids = patient_split(
        all_pids, args.train_ratio, args.val_ratio, args.seed
    )
    print(f"  Patients — train: {len(train_pids)} | val: {len(val_pids)} | test: {len(test_pids)}")

    # ── 3. Generate QA pairs ──────────────────────────────────────────────────
    # train: bucket by disease class for balancing
    # val/test: keep all (không balance để giữ representative test set)
    print("\n[3/4] Generating QA pairs...")

    train_buckets: dict[str, list] = defaultdict(list)
    val_samples:   list = []
    test_samples:  list = []
    stats = defaultdict(int)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Rows"):
        ma_ho_so = _clean(str(row.get(COL_MA_HO_SO, "")))
        if not ma_ho_so:
            stats["skip_no_id"] += 1
            continue

        split_name = (
            "train" if ma_ho_so in train_pids else
            "val"   if ma_ho_so in val_pids   else
            "test"
        )

        patient_dir = image_dir / ma_ho_so
        if not patient_dir.exists():
            stats["skip_no_folder"] += 1
            continue

        img_index = build_patient_image_index(patient_dir)
        if not img_index:
            stats["skip_no_folder"] += 1
            continue

        visits = expand_row_to_visits(row)
        if not visits:
            stats["skip_no_visits"] += 1
            continue

        stats["total_visits"] += len(visits)

        for visit in visits:
            filenames = visit.get("_filenames", [])
            if not filenames:
                stats["visit_no_files"] += 1
                continue

            dx_ans = make_diagnosis_answer(visit)
            tx_ans = make_treatment_answer(visit)
            if not dx_ans and not tx_ans:
                stats["visit_no_answer"] += 1
                continue

            for filename in filenames:
                abs_path = img_index.get(filename.lower())
                if abs_path is None:
                    stats["missing_image"] += 1
                    continue
                if not is_valid_image(str(abs_path)):
                    stats["invalid_image"] += 1
                    continue

                img_rel = str(abs_path.relative_to(image_dir))
                pairs   = make_qa_pairs(img_rel, visit)

                for p in pairs:
                    stats[f"qa_{p['qa_type']}"] += 1
                    stats["total_qa"] += 1

                    if split_name == "train":
                        # Tag with class for balancing (removed before saving)
                        p["_category"] = classify_answer(p.get("answer", ""))
                        train_buckets[p["_category"]].append(p)
                    elif split_name == "val":
                        val_samples.append(p)
                    else:
                        test_samples.append(p)

    total_train_raw = sum(len(v) for v in train_buckets.values())
    before_counts   = Counter({k: len(v) for k, v in train_buckets.items()})

    print(f"\n  Raw QA pairs — train: {total_train_raw} | val: {len(val_samples)} | test: {len(test_samples)}")
    print_distribution(before_counts, "BEFORE balancing (train split)", total_train_raw)

    # ── 4. Balance train split ─────────────────────────────────────────────────
    print("\n[4/4] Balancing train split...")
    train_balanced = apply_balance(
        train_buckets,
        max_per_class=args.max_per_class,
        max_total=args.max_total,
        min_keep=args.min_keep,
        seed=args.seed,
    )

    # Shuffle val/test too
    random.seed(args.seed)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    # ── 5. Save ────────────────────────────────────────────────────────────────
    suffix = args.output_suffix
    splits_out = {
        f"train_{suffix}": train_balanced,
        f"val_{suffix}":   val_samples,
        f"test_{suffix}":  test_samples,
    }

    print()
    for name, samples in splits_out.items():
        out_file = output_dir / f"dataset_{name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  {name:30s}: {len(samples):6d} QA pairs → {out_file}")

    # Stats file
    stats_file = output_dir / f"stats_{suffix}.txt"
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(f"=== Balanced Preprocess Stats (suffix={suffix}) ===\n\n")
        f.write(f"CSV rows              : {len(df)}\n")
        f.write(f"Total visits expanded : {stats['total_visits']}\n")
        f.write(f"Raw train QA          : {total_train_raw}\n")
        f.write(f"Balanced train QA     : {len(train_balanced)}\n")
        f.write(f"Val QA                : {len(val_samples)}\n")
        f.write(f"Test QA               : {len(test_samples)}\n")
        f.write(f"Skipped (no ID)       : {stats['skip_no_id']}\n")
        f.write(f"Skipped (no folder)   : {stats['skip_no_folder']}\n")
        f.write(f"Missing images        : {stats['missing_image']}\n")
        f.write(f"Total QA generated    : {stats['total_qa']}\n")
        f.write(f"  diagnosis           : {stats['qa_diagnosis']}\n")
        f.write(f"  treatment           : {stats['qa_treatment']}\n")
    print(f"\n  Stats → {stats_file}")

    print(f"\nDone!")
    print(f"\nNext — generate minimal context for train:")
    print(f"  python build_minimal_test.py \\")
    print(f"      --input  {output_dir}/dataset_train_{suffix}.jsonl \\")
    print(f"      --output {output_dir}/dataset_train_{suffix}_minimal.jsonl")


if __name__ == "__main__":
    main()
