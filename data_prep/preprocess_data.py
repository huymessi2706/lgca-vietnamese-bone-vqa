"""
Preprocessing Script: Clinical CSV + JPG images → JSONL for training
=====================================================================

Logic (Cách 1):
  - Group tất cả visits theo MaHoSo
  - Pick 1 "best record" per patient (inpatient+ICD > latest+ICD > latest)
  - Map TẤT CẢ ảnh của bệnh nhân vào best record đó
  - Chỉ giữ bệnh nhân trong qualified_patients.json (có ICD + XQ + ChanDoan + thuộc 5 nhóm)

Image naming convention:
  bone_data/CH240100172_0.jpg   (first image of patient CH240100172)
  bone_data/CH240100172_1.jpg   (second image, same patient)

Output:
  dataset_train.jsonl
  dataset_val.jsonl
  dataset_test.jsonl
  stats.txt

Usage:
  python preprocess_data.py \\
      --image_dir   ./bone_data \\
      --clinical_csv ./clinical_data/Copy\\ of\\ Data_TT_BenhNhan\\ -\\ Data_Full.csv \\
      --qualified    ./clinical_data/qualified_patients.json \\
      --output       ./data/dataset.jsonl \\
      --train_ratio  0.75 --val_ratio 0.10 --test_ratio 0.15
"""

import os
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Column mapping  CSV column name → internal key
# ---------------------------------------------------------------------------

COLUMN_MAP = {
    "patient_id":    "Mã hồ sơ",
    "admission_date":"Ngày khám (đầu)",
    "history":       "Quá trình bệnh lý",
    "past_history":  "Tiền sử bản thân",
    "local_exam":    "Khám các bộ phận",
    "lab_summary":   "Tóm tắt CLS",
    "initial_dx":    "Chẩn đoán vào viện",
    "initial_tx":    "Đã xử lý",
    "chief_complaint":"Lý do vào viện",
    "inpatient_day": "Số lần khám",
    "surgical_exam": "Khám toàn thân",
    "bone_exam":     "Cơ xương khớp",
    "lab_tests":     "Các XN cần làm",
    "main_dx":       "Chẩn đoán xác định",
    "comorbidity":   "Chẩn đoán vào khoa",
    "prognosis":     "Tiên lượng",
    "xray_findings": "Kết quả xét nghiệm",
    "surgery":       "Phương pháp điều trị",
    "final_dx":      "Chẩn đoán xác định",
    "icd_code":      "Mã ICD-10",
    "outcome":       "Tình trạng ra viện",
    "followup":      "Hướng điều trị tiếp theo",
    "image_filenames":"Tên file .jpg (tất cả)",
}

# Disease group labels (Vietnamese)
GROUP_LABEL_VI = {
    "u_xuong":     "U xương",
    "nhiem_trung": "Nhiễm trùng xương",
    "thoai_hoa":   "Thoái hóa xương",
    "chan_thuong":  "Chấn thương xương",
    "chuyen_hoa":  "Chuyển hóa xương",
}


# ---------------------------------------------------------------------------
# ICD → disease group
# ---------------------------------------------------------------------------

def classify_icd(icd: str):
    icd = icd.strip().upper()
    if not icd:
        return None
    if icd[0] in ("C", "D"):
        return "u_xuong"
    if any(icd.startswith(p) for p in ["M86", "M00", "M01", "M02", "M03", "M46", "A18"]):
        return "nhiem_trung"
    if any(icd.startswith(p) for p in [
        "M15", "M16", "M17", "M18", "M19",
        "M40", "M41", "M42", "M43", "M45",
        "M47", "M48", "M50", "M51", "M53", "M54",
    ]):
        return "thoai_hoa"
    if icd[0] in ("S", "T"):
        return "chan_thuong"
    if any(icd.startswith(p) for p in [
        "M80", "M81", "M82", "M83", "M84", "M85",
        "M90", "M91", "M92", "M93", "M94", "M95",
        "E55", "E21",
    ]):
        return "chuyen_hoa"
    return None


# ---------------------------------------------------------------------------
# Pick best record per patient
# ---------------------------------------------------------------------------

def pick_best_record(visits: list) -> dict:
    """
    Priority:
      1. Inpatient record (VaoNgayThu_BenhAn filled) WITH ICD
      2. Latest record WITH ICD
      3. Latest record (fallback)
    """
    inpatient_icd = [
        v for v in visits
        if v.get("inpatient_day", "").strip()
        and v.get("icd_code", "").strip()
    ]
    if inpatient_icd:
        return inpatient_icd[0]

    has_icd = [v for v in visits if v.get("icd_code", "").strip()]
    if has_icd:
        return sorted(has_icd, key=lambda x: x.get("admission_date", ""), reverse=True)[0]

    return sorted(visits, key=lambda x: x.get("admission_date", ""), reverse=True)[0]


# ---------------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------------

def load_clinical_data(csv_path: str) -> dict:
    """
    Returns dict: patient_id (str) → list of visit dicts (internal keys).
    """
    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig").fillna("")
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    patient_visits = defaultdict(list)
    for _, row in df.iterrows():
        visit = {key: row.get(col, "") for key, col in COLUMN_MAP.items()}
        pid = visit["patient_id"].strip().upper()
        if pid:
            patient_visits[pid].append(visit)

    print(f"  Loaded {len(df)} rows, {len(patient_visits)} unique patients")
    return patient_visits


# ---------------------------------------------------------------------------
# Build clinical context string
# ---------------------------------------------------------------------------

def format_clinical_context(v: dict) -> str:
    parts = []
    if v.get("chief_complaint"):
        parts.append(f"Lý do vào viện: {v['chief_complaint']}.")
    if v.get("history"):
        parts.append(f"Bệnh sử: {v['history']}.")
    if v.get("past_history") and v["past_history"] not in ("Chưa phát hiện bất thường", ""):
        parts.append(f"Tiền sử: {v['past_history']}.")
    if v.get("bone_exam"):
        parts.append(f"Cơ xương khớp: {v['bone_exam']}.")
    if v.get("local_exam"):
        parts.append(f"Tại chỗ: {v['local_exam']}.")
    if v.get("lab_summary"):
        parts.append(f"Cận lâm sàng: {v['lab_summary']}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Generate QA pairs from one (image, record) pair
# ---------------------------------------------------------------------------

def make_qa_pairs(image_rel_path: str, v: dict, disease_group) -> list:
    """
    disease_group: str (một trong 5 nhóm) hoặc None (không có ICD).
    Khi None, bỏ qua QA phân loại nhóm, vẫn sinh các QA còn lại.
    """
    context = format_clinical_context(v)
    pairs   = []

    # QA-1: Phân loại nhóm bệnh (chỉ khi có ICD → có disease_group)
    if disease_group is not None:
        group_label = GROUP_LABEL_VI.get(disease_group, disease_group)
        pairs.append({
            "image":         image_rel_path,
            "question":      (
                f"Thông tin lâm sàng: {context} "
                f"Dựa vào hình ảnh X-quang và thông tin lâm sàng, "
                f"bệnh nhân này thuộc nhóm bệnh xương nào trong các nhóm sau: "
                f"U xương, Nhiễm trùng xương, Thoái hóa xương, Chấn thương xương, Chuyển hóa xương?"
            ),
            "answer":        group_label,
            "disease_group": disease_group,
            "qa_type":       "classification",
        })

    # QA-2: Chẩn đoán xác định
    final_dx = v.get("final_dx") or v.get("initial_dx", "")
    if final_dx:
        pairs.append({
            "image":         image_rel_path,
            "question":      (
                f"Thông tin lâm sàng: {context} "
                f"Dựa vào hình ảnh X-quang và thông tin lâm sàng trên, "
                f"chẩn đoán xác định của bệnh nhân này là gì?"
            ),
            "answer":        final_dx,
            "disease_group": disease_group,
            "qa_type":       "diagnosis",
        })

    # QA-3: Mô tả X-quang
    xray = v.get("xray_findings", "")
    if xray:
        pairs.append({
            "image":         image_rel_path,
            "question":      (
                f"Thông tin lâm sàng: {context} "
                f"Mô tả những bất thường thấy được trên hình ảnh X-quang này."
            ),
            "answer":        xray,
            "disease_group": disease_group,
            "qa_type":       "xray_description",
        })

    # QA-4: Hướng điều trị
    tx = v.get("surgery") or v.get("initial_tx", "")
    if tx:
        pairs.append({
            "image":         image_rel_path,
            "question":      (
                f"Thông tin lâm sàng: {context} "
                f"Phương pháp điều trị hoặc phẫu thuật phù hợp cho bệnh nhân này là gì?"
            ),
            "answer":        tx,
            "disease_group": disease_group,
            "qa_type":       "treatment",
        })

    return pairs


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------

def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Patient-level train/val/test split
# ---------------------------------------------------------------------------

def patient_split(qualified_pids: list, train_r: float, val_r: float, seed: int):
    """
    Split patient IDs at patient level (not sample level) to avoid leakage.
    Returns (train_pids, val_pids, test_pids) as sets.
    """
    random.seed(seed)
    pids = [q["patient_id"] for q in qualified_pids]
    random.shuffle(pids)
    n     = len(pids)
    n_tr  = int(n * train_r)
    n_val = int(n * val_r)
    return (
        set(pids[:n_tr]),
        set(pids[n_tr: n_tr + n_val]),
        set(pids[n_tr + n_val:]),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",    default="./bone_data",
                        help="Folder chứa ảnh JPG (naming: PATIENTID_N.jpg)")
    parser.add_argument("--clinical_csv", default="./clinical_data/Copy of Data_TT_BenhNhan - Data_Full.csv")
    parser.add_argument("--qualified",    default=None,
                        help="File JSON danh sách bệnh nhân đủ điều kiện. "
                             "Nếu không truyền → dùng toàn bộ bệnh nhân trong CSV.")
    parser.add_argument("--output",       default="./data/dataset.jsonl")
    parser.add_argument("--train_ratio",  type=float, default=0.75)
    parser.add_argument("--val_ratio",    type=float, default=0.10)
    parser.add_argument("--test_ratio",   type=float, default=0.15)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "train + val + test ratios must sum to 1.0"

    # ------------------------------------------------------------------
    # 1. Load qualified patient list (optional)
    # ------------------------------------------------------------------
    if args.qualified:
        print("\n[1/5] Đọc danh sách bệnh nhân đủ điều kiện...")
        with open(args.qualified, encoding="utf-8") as f:
            qualified = json.load(f)
        qualified_map = {q["patient_id"].upper(): q for q in qualified}
        print(f"  Qualified patients (from file): {len(qualified_map)}")
    else:
        print("\n[1/5] Không có file qualified → dùng toàn bộ bệnh nhân trong CSV.")
        qualified_map = None  # signal: use all

    # ------------------------------------------------------------------
    # 2. Load clinical data, group by patient
    # ------------------------------------------------------------------
    print("\n[2/5] Đọc dữ liệu lâm sàng...")
    patient_visits = load_clinical_data(args.clinical_csv)

    # Build best-record index
    best_records = {}
    if qualified_map is not None:
        # Mode A: whitelist từ file qualified
        for pid, meta in qualified_map.items():
            visits = patient_visits.get(pid, [])
            if not visits:
                continue
            best_records[pid] = (pick_best_record(visits), meta["disease_group"])
    else:
        # Mode B: toàn bộ bệnh nhân — classify ICD inline, group=None nếu không có ICD
        for pid, visits in patient_visits.items():
            best = pick_best_record(visits)
            icd  = best.get("icd_code", "").strip()
            grp  = classify_icd(icd) if icd else None
            # Cần ít nhất XQ hoặc ChanDoan để sinh QA có nghĩa
            has_content = (
                best.get("xray_findings", "").strip()
                or best.get("final_dx", "").strip()
                or best.get("initial_dx", "").strip()
            )
            if not has_content:
                continue
            best_records[pid] = (best, grp)
    print(f"  Best records built: {len(best_records)}")

    # ------------------------------------------------------------------
    # 3. Build patient_images from CSV (read filenames from clinical data)
    # ------------------------------------------------------------------
    print("\n[3/5] Lập chỉ mục ảnh từ CSV...")
    patient_images = defaultdict(list)   # PATIENTID → [abs_path, ...]
    total_images = 0

    for pid, (best, disease_group) in best_records.items():
        img_filenames_str = best.get("image_filenames", "").strip()
        if not img_filenames_str:
            continue

        # Split by newline (CSV can have multi-line cells)
        filenames = [f.strip() for f in img_filenames_str.split('\n') if f.strip()]

        for fname in filenames:
            # Build full path: search in bone_data recursively
            img_path = None
            for candidate in Path(args.image_dir).rglob(fname):
                if candidate.is_file():
                    img_path = str(candidate)
                    break

            if img_path:
                patient_images[pid].append(img_path)
                total_images += 1

    print(f"  Tổng ảnh tìm thấy: {total_images}")

    # ------------------------------------------------------------------
    # 4. Patient-level split
    # ------------------------------------------------------------------
    print("\n[4/5] Chia train/val/test theo bệnh nhân...")
    # Dùng danh sách PID từ best_records (đã lọc có content)
    all_qualified = [{"patient_id": pid} for pid in best_records]
    train_pids, val_pids, test_pids = patient_split(
        all_qualified, args.train_ratio, args.val_ratio, args.seed
    )
    print(f"  Train: {len(train_pids)} | Val: {len(val_pids)} | Test: {len(test_pids)}")

    # ------------------------------------------------------------------
    # 5. Generate QA pairs
    # ------------------------------------------------------------------
    print("\n[5/5] Tạo QA pairs...")

    splits      = {"train": [], "val": [], "test": []}
    stats       = {
        "total_patients_with_images": 0,
        "no_image": 0,
        "invalid_image": 0,
        "total_qa_pairs": 0,
        "group_counts": defaultdict(int),
        "qa_type_counts": defaultdict(int),
    }

    for pid, (best, disease_group) in tqdm(best_records.items(), desc="  Processing"):
        images = patient_images.get(pid, [])

        if not images:
            stats["no_image"] += 1
            continue

        stats["total_patients_with_images"] += 1

        # Determine split
        if pid in train_pids:
            split_name = "train"
        elif pid in val_pids:
            split_name = "val"
        else:
            split_name = "test"

        for abs_path in images:
            if not is_valid_image(abs_path):
                stats["invalid_image"] += 1
                continue

            img_rel = os.path.relpath(abs_path, args.image_dir)
            pairs   = make_qa_pairs(img_rel, best, disease_group)

            splits[split_name].extend(pairs)
            stats["total_qa_pairs"] += len(pairs)
            for p in pairs:
                stats["group_counts"][disease_group] += 1
                stats["qa_type_counts"][p["qa_type"]] += 1

    # ------------------------------------------------------------------
    # Save JSONL
    # ------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for split_name, samples in splits.items():
        random.shuffle(samples)
        out_file = out_path.parent / f"{out_path.stem}_{split_name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(samples)} QA pairs → {out_file}")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    stats_path = out_path.parent / "stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"Total qualified patients : {len(best_records)}\n")
        f.write(f"Patients with images     : {stats['total_patients_with_images']}\n")
        f.write(f"Patients without images  : {stats['no_image']}\n")
        f.write(f"Invalid images skipped   : {stats['invalid_image']}\n")
        f.write(f"Total QA pairs           : {stats['total_qa_pairs']}\n\n")
        f.write("Group distribution:\n")
        for g, cnt in sorted(stats["group_counts"].items(), key=lambda x: -x[1]):
            f.write(f"  {g}: {cnt}\n")
        f.write("\nQA type distribution:\n")
        for t, cnt in sorted(stats["qa_type_counts"].items(), key=lambda x: -x[1]):
            f.write(f"  {t}: {cnt}\n")
        f.write(f"\nSplit sizes:\n")
        for sn, samples in splits.items():
            f.write(f"  {sn}: {len(samples)}\n")

    print(f"\n  Stats saved to {stats_path}")
    print(f"\nDone! Total QA pairs: {stats['total_qa_pairs']}")
    print("Group distribution:")
    for g, cnt in sorted(stats["group_counts"].items(), key=lambda x: -x[1]):
        print(f"  {g}: {cnt}")


if __name__ == "__main__":
    main()
