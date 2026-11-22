"""
Preprocessing Script (no-diacritics CSV version)
=================================================
Giống preprocess_data.py nhưng dùng đúng tên cột thực tế trong CSV
(không dấu) và tự scan ảnh theo pattern {MaHoSo}_*.jpg thay vì đọc
từ cột image_filenames (không tồn tại trong CSV hiện tại).

Usage:
  python preprocess_test.py \
      --image_dir   ./bone_data \
      --clinical_csv "./clinical_data/Copy of Data_TT_BenhNhan - Data_Full.csv" \
      --output       ./data/dataset.jsonl \
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
# Column mapping — tên cột thực tế trong CSV (không dấu)
# ---------------------------------------------------------------------------

COLUMN_MAP = {
    "patient_id":     "MaHoSo",
    "admission_date": "NgayKham",
    "history":        "QuaTrinhBenhLy_BenhAn",
    "past_history":   "TienSuBenh_BanThan_BenhAn",
    "local_exam":     "KhamXet_CacBoPhan",
    "lab_summary":    "KhamXet_TomTatCanLamSang",
    "initial_dx":     "KhamXet_ChanDoanVaoVien",
    "initial_tx":     "KhamXet_DaXuLi",
    "chief_complaint":"LyDoVaoVien_BenhAN",
    "inpatient_day":  "VaoNgayThu_BenhAn",
    "surgical_exam":  "KhamBenh_NgoaiKhoa",
    "bone_exam":      "CacCoQuan_CoXuongKhop",
    "lab_tests":      "CacXetNghiemCanLam",
    "main_dx":        "ChanDoanKhiVaoKhoaDieuTri_BenhChinh",
    "comorbidity":    "ChanDoanKhiVaoKhoaDieuTri_BenhKemTheo",
    "prognosis":      "Tiên Lượng",
    "xray_findings":  "TongKetBenhAn_TomTatKetQuaXetNghiem",
    "surgery":        "TongKetBenhAn_PhuongPhapDieuTri",
    "final_dx":       "TongKetBenhAn_ChanDoanXacDinh",
    "icd_code":       "TongKetBenhAn_MaICD_Chinh",
    "outcome":        "TongKetBenhAn_TinhTrangNguoiBenhRaVien",
    "followup":       "TongKetBenhAn_HuongDieuTriVaCacCheDoTiepTheo",
}

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
    inpatient_icd = [
        v for v in visits
        if v.get("inpatient_day", "").strip() and v.get("icd_code", "").strip()
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
    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig").fillna("")
    # Pandas renames duplicate columns KhamBenh_ToanThan → KhamBenh_ToanThan.1
    # We only use KhamBenh_NgoaiKhoa so no conflict.
    df.columns = [c.strip() for c in df.columns]
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
# Generate QA pairs
# ---------------------------------------------------------------------------

def make_qa_pairs(image_rel_path: str, v: dict, disease_group) -> list:
    context = format_clinical_context(v)
    pairs   = []

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

def patient_split(pids: list, train_r: float, val_r: float, seed: int):
    random.seed(seed)
    pids = list(pids)
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
    parser.add_argument("--image_dir",    default="./bone_data")
    parser.add_argument("--clinical_csv", default="./clinical_data/Copy of Data_TT_BenhNhan - Data_Full.csv")
    parser.add_argument("--output",       default="./data/dataset.jsonl")
    parser.add_argument("--train_ratio",  type=float, default=0.75)
    parser.add_argument("--val_ratio",    type=float, default=0.10)
    parser.add_argument("--test_ratio",   type=float, default=0.15)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "train + val + test ratios must sum to 1.0"

    # ------------------------------------------------------------------
    # 1. Load clinical data
    # ------------------------------------------------------------------
    print("\n[1/4] Đọc dữ liệu lâm sàng...")
    patient_visits = load_clinical_data(args.clinical_csv)

    best_records = {}
    for pid, visits in patient_visits.items():
        best = pick_best_record(visits)
        icd  = best.get("icd_code", "").strip()
        grp  = classify_icd(icd) if icd else None
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
    # 2. Scan images by patient ID pattern: {PID}_*.jpg
    # ------------------------------------------------------------------
    print("\n[2/4] Scan ảnh theo pattern {{PID}}_*.jpg ...")
    image_dir = Path(args.image_dir)
    patient_images = defaultdict(list)
    total_found = 0

    for pid in best_records:
        # Case-insensitive: try both upper and lower just in case
        matches = list(image_dir.rglob(f"{pid}_*.jpg")) + \
                  list(image_dir.rglob(f"{pid}_*.JPG"))
        for p in matches:
            patient_images[pid].append(str(p))
            total_found += 1

    print(f"  Tổng ảnh tìm thấy: {total_found} từ {len(patient_images)} bệnh nhân")

    # ------------------------------------------------------------------
    # 3. Patient-level split
    # ------------------------------------------------------------------
    print("\n[3/4] Chia train/val/test theo bệnh nhân...")
    train_pids, val_pids, test_pids = patient_split(
        list(best_records.keys()), args.train_ratio, args.val_ratio, args.seed
    )
    print(f"  Train: {len(train_pids)} | Val: {len(val_pids)} | Test: {len(test_pids)}")

    # ------------------------------------------------------------------
    # 4. Generate QA pairs
    # ------------------------------------------------------------------
    print("\n[4/4] Tạo QA pairs...")
    splits = {"train": [], "val": [], "test": []}
    stats  = {
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
        split_name = "train" if pid in train_pids else ("val" if pid in val_pids else "test")

        for abs_path in images:
            if not is_valid_image(abs_path):
                stats["invalid_image"] += 1
                continue

            img_rel = os.path.relpath(abs_path, str(image_dir))
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

    # Stats
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
        f.write("\nSplit sizes:\n")
        for sn, s in splits.items():
            f.write(f"  {sn}: {len(s)}\n")

    print(f"\n  Stats → {stats_path}")
    print(f"\nDone! Total QA pairs: {stats['total_qa_pairs']}")
    for g, cnt in sorted(stats["group_counts"].items(), key=lambda x: -x[1]):
        print(f"  {g}: {cnt}")


if __name__ == "__main__":
    main()
