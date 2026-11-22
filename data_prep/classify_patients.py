"""
classify_patients.py
====================
Phân loại bệnh nhân vào 5 nhóm bệnh xương dựa trên
`TongKetBenhAn_ChanDoanXacDinh` (chẩn đoán xác định của bác sĩ).

OUTPUT:
  - classified_patients.csv  : MaHoSo, NgayKham, ChanDoan, group, confidence, ambiguous
  - group_summary.csv        : thống kê số BN theo nhóm
  - ambiguous_review.csv     : các ca cần review tay (score gần bằng nhau)
  - unclassified.csv         : ca không khớp từ khóa nào

LOGIC:
  - Mỗi BN có thể có nhiều lần khám → xét TẤT CẢ các lần
  - Score keyword cho mỗi nhóm → chọn nhóm có score cao nhất
  - Ưu tiên: U_XUONG > NHIEM_TRUNG > XUONG_CHUYEN_HOA > THOAI_HOA > CHAN_THUONG
  - "Gãy bệnh lý" do u/loãng xương → gán theo bệnh nền, không gán CHAN_THUONG
"""

import csv
import json
import re
import os
from collections import defaultdict
from typing import Optional, List, Dict

# ─────────────────────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────────────────────
CSV_PATH   = "./clinical_data/Copy of Data_TT_BenhNhan - Data_Full.csv"
OUTPUT_DIR = "./clinical_data/classified/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cột chẩn đoán chính để phân loại
DIAG_COLS = [
    "TongKetBenhAn_ChanDoanXacDinh",      # chẩn đoán xác định (CHÍNH)
    "ChanDoanKhiVaoKhoaDieuTri_BenhChinh", # chẩn đoán vào khoa (phụ)
    "TongKetBenhAn_MaICD_Chinh",           # mã ICD (phụ, để cross-check)
]

# ─────────────────────────────────────────────────────────────
# KEYWORD PATTERNS — (pattern, score)
# score cao hơn = từ khóa đặc hiệu hơn
# ─────────────────────────────────────────────────────────────
KEYWORDS = {
    "U_XUONG": [
        # Ác tính
        (r"sarcom",                     10),
        (r"sarcoma",                    10),
        (r"osteosarcoma",               10),
        (r"ung thư xương",              10),
        (r"di căn xương",               10),
        (r"ung thư",                     8),
        (r"ác tính",                     8),
        (r"lymphoma",                    8),
        (r"myeloma",                     8),
        (r"leuk[eê]mi",                  8),
        # Lành tính / ranh giới
        (r"u xương",                     9),
        (r"u sụn",                       9),
        (r"u lành",                      7),
        (r"bướu xương",                  9),
        (r"bướu lành",                   7),
        (r"bướu ác",                     9),
        (r"bọc xương",                   8),   # aneurysmal bone cyst, simple cyst
        (r"nang xương",                  8),
        (r"loạn sản sợi",                8),   # fibrous dysplasia
        (r"loạn sản",                    6),
        (r"u tế bào khổng lồ",           9),
        (r"giant cell",                  9),
        (r"u mạch",                      7),
        (r"u mỡ",                        7),
        (r"u thần kinh",                 7),
        (r"chondroma",                   9),
        (r"osteoma",                     9),
        (r"enchondroma",                 9),
        (r"exostosis",                   8),
        (r"osteochondroma",              8),
        (r"sinh thiết",                  3),   # yếu — thường đi kèm u
        (r"cắt nạo bướu",               5),
        (r"cắt rộng bướu",              6),
        (r"hóa trị",                     5),
        (r"xạ trị",                      5),
        (r"gãy bệnh lý.{0,40}(u|bướu|loạn sản|sarcom)", 9),  # gãy bệnh lý do u
    ],

    "NHIEM_TRUNG_XUONG": [
        (r"nhiễm trùng xương",          10),
        (r"viêm xương",                 10),
        (r"viêm tủy xương",             10),
        (r"osteomyelitis",              10),
        (r"viêm khớp nhiễm",             9),
        (r"nhiễm trùng khớp",            9),
        (r"áp xe xương",                 9),
        (r"áp xe",                       6),
        (r"mủ xương",                    9),
        (r"mủ khớp",                     9),
        (r"nhiễm trùng vết mổ",          7),
        (r"nhiễm khuẩn xương",           9),
        (r"hoại tử xương",               7),   # có thể do nhiễm trùng
        (r"septic arthritis",            9),
        (r"lao xương",                   9),
        (r"lao khớp",                    9),
        (r"brucella",                    9),
        (r"xi măng kháng sinh",          5),   # dùng khi nhiễm trùng xương
        (r"cấy m[ủu]",                   4),   # cấy mủ kháng sinh đồ
    ],

    "THOAI_HOA_XUONG": [
        (r"thoái hóa",                  10),
        (r"thoái hoá",                  10),
        (r"vẹo cột sống",                9),
        (r"trượt đốt sống",              9),
        (r"gai cột sống",                8),
        (r"thoát vị đĩa đệm",            9),
        (r"hẹp ống sống",                8),
        (r"viêm khớp dạng thấp",         9),
        (r"viêm khớp",                   6),
        (r"hội chứng ống cổ tay",        7),
        (r"hội chứng đường hầm",         7),
        (r"gout",                        7),
        (r"gút",                         7),
        (r"calci hóa",                   5),
        (r"xơ cứng",                     5),
        (r"rỗng tủy",                    7),   # syringomyelia — thường kèm vẹo cột sống
        (r"skoliosis",                   8),
        (r"scoliosis",                   8),
        (r"kyphosis",                    8),
        (r"gù cột sống",                 8),
        (r"vẹo",                         4),
        (r"trật đốt sống",               7),
        (r"bệnh đĩa đệm",                8),
        (r"thoát vị nhân nhầy",          8),
    ],

    "XUONG_CHUYEN_HOA": [
        (r"loãng xương",                10),
        (r"loang xương",                10),   # typo variant
        (r"còi xương",                  10),
        (r"nhuyễn xương",               10),
        (r"osteoporosis",               10),
        (r"osteomalacia",               10),
        (r"rickets",                    10),
        (r"cường cận giáp",             10),
        (r"suy cận giáp",               10),
        (r"paget",                       9),
        (r"bệnh xương chuyển hóa",      10),
        (r"rối loạn chuyển hóa xương",  10),
        (r"thiếu vitamin d",             8),
        (r"hạ calci",                    7),
        (r"tăng calci",                  6),
        (r"phosphat",                    5),
        (r"fluorosis",                   9),
        (r"bệnh xương do thận",          9),
        (r"gãy bệnh lý.{0,40}loãng xương", 9),  # gãy do loãng xương
    ],

    "CHAN_THUONG_XUONG": [
        (r"gãy",                         4),   # rất phổ biến → score thấp
        (r"gay",                         3),   # typo
        (r"trật khớp",                   5),
        (r"vỡ",                          4),
        (r"lún",                         5),
        (r"đứt gân",                     6),
        (r"rách gân",                    6),
        (r"đứt dây chằng",               6),
        (r"rách dây chằng",              6),
        (r"chấn thương",                 4),
        (r"tai nạn",                     3),
        (r"té ngã",                      3),
        (r"bong gân",                    5),
        (r"hematoma",                    4),
        (r"tụ máu",                      4),
        (r"kết hợp xương",               3),
        (r"nẹp vít",                     3),
        (r"đinh nội tủy",                4),
        (r"ilizarov",                    4),
        (r"vết thương",                  3),
        (r"lộ xương",                    4),
        (r"cẳng chân",                   2),
        (r"xương đùi",                   2),
        (r"xương mác",                   2),
        (r"xương chày",                  2),
    ],
}

# ICD prefix mapping (cross-check phụ)
ICD_PREFIX_GROUP = {
    "C": "U_XUONG",        # Malignant neoplasm
    "D1": "U_XUONG",       # Benign neoplasm of bone
    "D16": "U_XUONG",
    "M8": "U_XUONG",       # Bone tumors in ICD
    "M9": "NHIEM_TRUNG_XUONG",
    "M86": "NHIEM_TRUNG_XUONG",  # Osteomyelitis
    "M00": "NHIEM_TRUNG_XUONG",  # Infectious arthropathy
    "M40": "THOAI_HOA_XUONG",
    "M41": "THOAI_HOA_XUONG",  # Scoliosis
    "M47": "THOAI_HOA_XUONG",
    "M48": "THOAI_HOA_XUONG",
    "M80": "XUONG_CHUYEN_HOA",  # Osteoporosis with fracture
    "M81": "XUONG_CHUYEN_HOA",  # Osteoporosis without fracture
    "M83": "XUONG_CHUYEN_HOA",  # Adult osteomalacia
    "S": "CHAN_THUONG_XUONG",   # Injuries
    "T": "CHAN_THUONG_XUONG",   # Injuries/poisoning
}

PRIORITY = ["U_XUONG", "NHIEM_TRUNG_XUONG", "XUONG_CHUYEN_HOA", "THOAI_HOA_XUONG", "CHAN_THUONG_XUONG"]

GROUP_LABEL_VI = {
    "U_XUONG":           "U xương (Bone Tumor)",
    "NHIEM_TRUNG_XUONG": "Nhiễm trùng xương (Osteomyelitis)",
    "CHAN_THUONG_XUONG":  "Gãy xương / Chấn thương (Fracture/Trauma)",
    "THOAI_HOA_XUONG":   "Bệnh xương thoái hóa (Degenerative)",
    "XUONG_CHUYEN_HOA":  "Bệnh xương chuyển hóa (Metabolic)",
}


# ─────────────────────────────────────────────────────────────
# SCORE FUNCTION
# ─────────────────────────────────────────────────────────────
def score_text(text: str) -> dict:
    """Tính score cho từng nhóm dựa trên keyword matching."""
    text_lower = text.lower()
    scores = {g: 0 for g in KEYWORDS}

    for group, patterns in KEYWORDS.items():
        for pattern, weight in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                # Cộng dồn nhưng giới hạn để tránh 1 từ xuất hiện nhiều lần inflate score
                scores[group] += weight * min(len(matches), 3)

    return scores


def icd_hint(icd_code: str) -> Optional[str]:
    """Tra mã ICD → gợi ý nhóm (phụ)."""
    if not icd_code:
        return None
    icd = icd_code.strip().upper()
    for prefix, group in ICD_PREFIX_GROUP.items():
        if icd.startswith(prefix):
            return group
    return None


def classify_patient(visits: List[Dict]) -> dict:
    """
    Phân loại 1 bệnh nhân từ tất cả lần khám.
    Returns: {group, confidence, ambiguous, scores, all_diag}
    """
    combined_text = ""
    all_diag = []

    for visit in visits:
        parts = []
        for col in DIAG_COLS:
            val = visit.get(col, "")
            if val and val.strip().lower() not in {"nan", "", "không", "none"}:
                parts.append(val.strip())
        visit_text = " | ".join(parts)
        combined_text += " " + visit_text
        if parts:
            all_diag.append(visit_text)

    # Score từ text
    scores = score_text(combined_text)

    # ICD hint từ tất cả visits
    for visit in visits:
        icd = visit.get("TongKetBenhAn_MaICD_Chinh", "")
        hint = icd_hint(icd)
        if hint:
            scores[hint] += 3  # boost nhẹ từ ICD

    # Chọn nhóm
    max_score = max(scores.values())

    if max_score == 0:
        return {
            "group": None,
            "confidence": 0,
            "ambiguous": False,
            "scores": scores,
            "all_diag": " || ".join(all_diag),
        }

    # Các nhóm có score cao (trong khoảng 80% max)
    top_groups = [g for g, s in scores.items() if s >= max_score * 0.8 and s > 0]

    if len(top_groups) == 1:
        chosen = top_groups[0]
        ambiguous = False
    else:
        # Nhiều nhóm cạnh tranh → dùng priority để tie-break
        chosen = next((g for g in PRIORITY if g in top_groups), top_groups[0])
        ambiguous = True

    confidence = round(scores[chosen] / (sum(scores.values()) + 1e-9), 3)

    return {
        "group": chosen,
        "confidence": confidence,
        "ambiguous": ambiguous,
        "scores": scores,
        "all_diag": " || ".join(all_diag[:3]),  # giữ tối đa 3 lần đầu
    }


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def run():
    print("=" * 65)
    print("PHÂN LOẠI BỆNH NHÂN THEO 5 NHÓM BỆNH XƯƠNG")
    print("Dựa trên: TongKetBenhAn_ChanDoanXacDinh (chẩn đoán bác sĩ)")
    print("=" * 65)

    # 1. Đọc CSV
    print(f"\n[1/3] Đọc CSV: {CSV_PATH}")
    rows = []
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"   Tổng dòng: {len(rows):,}")
    unique_ids = set(r["MaHoSo"].strip() for r in rows)
    print(f"   Bệnh nhân unique: {len(unique_ids):,}")

    # Group rows by MaHoSo
    by_patient = defaultdict(list)
    for row in rows:
        maho = row["MaHoSo"].strip()
        by_patient[maho].append(row)

    # 2. Phân loại
    print(f"\n[2/3] Phân loại {len(by_patient):,} bệnh nhân...")
    results = []
    group_counts = defaultdict(int)
    ambiguous_count = 0
    unclassified_count = 0

    for maho, visits in by_patient.items():
        # Lấy NgayKham mới nhất
        dates = sorted(
            [v.get("NgayKham", "") for v in visits],
            reverse=True
        )
        latest_date = dates[0] if dates else ""

        result = classify_patient(visits)

        if result["group"] is None:
            unclassified_count += 1
        else:
            group_counts[result["group"]] += 1

        if result["ambiguous"]:
            ambiguous_count += 1

        results.append({
            "MaHoSo":     maho,
            "NgayKham_Latest": latest_date,
            "n_visits":   len(visits),
            "group":      result["group"] or "UNCLASSIFIED",
            "confidence": result["confidence"],
            "ambiguous":  result["ambiguous"],
            "score_U_XUONG":          result["scores"]["U_XUONG"],
            "score_NHIEM_TRUNG":      result["scores"]["NHIEM_TRUNG_XUONG"],
            "score_THOAI_HOA":        result["scores"]["THOAI_HOA_XUONG"],
            "score_XUONG_CHUYEN_HOA": result["scores"]["XUONG_CHUYEN_HOA"],
            "score_CHAN_THUONG":       result["scores"]["CHAN_THUONG_XUONG"],
            "ChanDoan_summary":       result["all_diag"][:300],
        })

    # 3. Ghi output
    print(f"\n[3/3] Ghi kết quả...")

    fieldnames = list(results[0].keys())

    # 3a. Tất cả kết quả
    all_path = os.path.join(OUTPUT_DIR, "classified_patients.csv")
    with open(all_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"   ✓ {all_path}  ({len(results):,} bệnh nhân)")

    # 3b. Ambiguous
    amb_path = os.path.join(OUTPUT_DIR, "ambiguous_review.csv")
    amb_rows = [r for r in results if r["ambiguous"] and r["group"] != "UNCLASSIFIED"]
    with open(amb_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(amb_rows)
    print(f"   ✓ {amb_path}  ({len(amb_rows):,} ca cần review)")

    # 3c. Unclassified
    unc_path = os.path.join(OUTPUT_DIR, "unclassified.csv")
    unc_rows = [r for r in results if r["group"] == "UNCLASSIFIED"]
    with open(unc_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unc_rows)
    print(f"   ✓ {unc_path}  ({len(unc_rows):,} ca không xác định được)")

    # 3d. File mahs_theo_5_nhom (dùng trực tiếp cho training pipeline)
    groups_path = os.path.join(OUTPUT_DIR, "mahs_theo_5_nhom.csv")
    with open(groups_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["MaHoSo", "group"])
        writer.writeheader()
        for r in results:
            if r["group"] != "UNCLASSIFIED":
                writer.writerow({"MaHoSo": r["MaHoSo"], "group": r["group"]})
    print(f"   ✓ {groups_path}  ← dùng cho training pipeline")

    # 4. Thống kê
    print("\n" + "=" * 65)
    print("KẾT QUẢ PHÂN LOẠI")
    print("=" * 65)
    total_classified = sum(group_counts.values())
    for g in PRIORITY:
        cnt = group_counts.get(g, 0)
        pct = cnt / total_classified * 100 if total_classified else 0
        bar = "█" * (cnt // max(total_classified // 40, 1))
        print(f"  {g:<25}: {cnt:>5} ({pct:5.1f}%)  {bar}")
        print(f"    → {GROUP_LABEL_VI[g]}")

    print(f"\n  Tổng có nhóm     : {total_classified:,}")
    print(f"  Ambiguous (2 nhóm): {ambiguous_count:,}")
    print(f"  Không phân loại  : {unclassified_count:,}")
    print("=" * 65)
    print(f"\n  Output folder: {OUTPUT_DIR}")
    print("  File dùng cho training: mahs_theo_5_nhom.csv")


if __name__ == "__main__":
    run()
