"""
Preprocess Enhanced — New folder/CSV structure with multi-visit support
=======================================================================
Folder ảnh:
  bone_data/{Mã hồ sơ}/{ngày khám}/{image.jpg}

CSV: Ho_So_Lam_Sang_Day_Du.csv
  - Row có Số lần khám = 1 : data thuần, không có separator
  - Row có Số lần khám > 1 : data được gộp vào 1 row với format lồng nhau:
      Images  → "── DD/MM/YYYY ──\nfile1.jpg\n── DD/MM/YYYY ──\nfile2.jpg"
      Clinical→ "● [LẦN KHÁM 1 | DD/MM/YYYY | Ca: XXXXX]\ncontent\n────\n● [LẦN KHÁM 2...]"

Script tự động expand multi-visit rows thành từng visit record riêng lẻ,
sau đó tạo 2 loại QA per ảnh: diagnosis + treatment.

Usage:
  python preprocess_enhanced.py \\
      --image_dir   ./bone_data \\
      --clinical_csv ./Ho_So_Lam_Sang_Day_Du.csv \\
      --output_dir   ./data_enhanced \\
      --train_ratio  0.75 --val_ratio 0.10 --test_ratio 0.15
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------

COL_MA_HO_SO    = "Mã hồ sơ"
COL_MA_BN       = "Mã bệnh nhân"
COL_SO_LAN      = "Số lần khám"
COL_NGAY_DAU    = "Ngày khám (đầu)"
COL_TEN_FILE    = "Tên file .jpg (tất cả)"
COL_LY_DO       = "Lý do vào viện"
COL_QUA_TRINH   = "Quá trình bệnh lý"
COL_TIEN_SU     = "Tiền sử bản thân"
COL_KHAM_BP     = "Khám các bộ phận"
COL_TOAN_THAN   = "Khám toàn thân"
COL_CO_XUONG    = "Cơ xương khớp"
COL_TOM_TAT     = "Tóm tắt CLS"
COL_KQ_XN       = "Kết quả xét nghiệm"
COL_CHAN_DOAN   = "Chẩn đoán xác định"
COL_PP_DT       = "Phương pháp điều trị"
COL_HUONG_DT    = "Hướng điều trị tiếp theo"
COL_ICD         = "Mã ICD-10"

# Clinical columns that may contain per-visit sections
CLINICAL_COLS = [
    COL_LY_DO, COL_QUA_TRINH, COL_TIEN_SU, COL_KHAM_BP,
    COL_TOAN_THAN, COL_CO_XUONG, COL_TOM_TAT,
    COL_KQ_XN, COL_CHAN_DOAN, COL_PP_DT, COL_HUONG_DT,
]

# Tokens that are not informative as answers
NOISE_TOKENS = {
    "xq", "xquang", "x-quang", "không", "khong", "chưa", "chua",
    "toa", "thuoc", "thuốc", "nội khoa", "ngoại khoa", "chưa", "không",
}

# ---------------------------------------------------------------------------
# Regex patterns for multi-visit detection and splitting
# ---------------------------------------------------------------------------

# Image date separator: "── 04/01/2023 ──"
IMG_DATE_SEP_RE = re.compile(r'──\s*(\d{1,2}/\d{1,2}/\d{4})\s*──')

# Clinical visit header: "● [LẦN KHÁM 1  |  04/01/2023  |  Ca: 102275]"
VISIT_HDR_RE = re.compile(
    r'●\s*\[LẦN KHÁM\s*(\d+)\s*\|\s*(\d{1,2}/\d{1,2}/\d{4})\s*\|[^\]]*\]'
)

# Long dash divider between visits in clinical columns
DIVIDER_RE = re.compile(r'^─{10,}$', re.MULTILINE)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _clean(val) -> str:
    if not isinstance(val, str):
        return ""
    v = val.strip()
    return "" if v.lower() in {"nan", "-", "—", ""} else v


def _is_noise(val: str) -> bool:
    v = val.strip().lower()
    return v in NOISE_TOKENS or len(v) <= 2


def _is_multi_visit(row: pd.Series) -> bool:
    """Detect if this row contains embedded multi-visit data."""
    # Check image column for date separators
    img_raw = _clean(row.get(COL_TEN_FILE, ""))
    if IMG_DATE_SEP_RE.search(img_raw):
        return True
    # Check any clinical column for visit headers
    for col in CLINICAL_COLS:
        if VISIT_HDR_RE.search(_clean(row.get(col, ""))):
            return True
    return False


def parse_image_visits(raw: str) -> list:
    """
    Parse the 'Tên file .jpg' cell for multi-visit rows.
    Returns list of (date_str, [filename, ...]) per visit.

    Single-visit (no ── markers): returns [("", [file1, file2, ...])]
    Multi-visit:
      "── 04/01/2023 ──\nimg1.jpg\nimg2.jpg\n── 29/03/2023 ──\nimg3.jpg"
      → [("04/01/2023", ["img1.jpg", "img2.jpg"]), ("29/03/2023", ["img3.jpg"])]
    """
    if not raw.strip():
        return []

    parts = IMG_DATE_SEP_RE.split(raw)
    # split returns: ['', date1, text1, date2, text2, ...]
    #  or ['text'] if no markers

    if len(parts) == 1:
        # Single visit
        files = [f.strip() for f in raw.splitlines()
                 if f.strip() and not IMG_DATE_SEP_RE.match(f.strip())]
        return [("", files)]

    visits = []
    for i in range(1, len(parts), 2):
        date = parts[i].strip()
        block = parts[i + 1] if i + 1 < len(parts) else ""
        files = [f.strip() for f in block.splitlines()
                 if f.strip() and not IMG_DATE_SEP_RE.match(f.strip())]
        visits.append((date, files))
    return visits


def parse_clinical_visits(raw: str) -> list:
    """
    Parse a clinical column cell for multi-visit rows.
    Returns list of (date_str, content_str) per visit.

    Single-visit (no ● markers): returns [("", raw_text)]
    Multi-visit:
      "● [LẦN KHÁM 1 | 04/01/2023 | Ca: X]\ncontent1\n─────\n● [LẦN KHÁM 2 | 29/03/2023 | Ca: Y]\ncontent2"
      → [("04/01/2023", "content1"), ("29/03/2023", "content2")]
    """
    if not raw.strip():
        return [("", "")]

    if not VISIT_HDR_RE.search(raw):
        return [("", raw.strip())]

    # Split on visit headers; capturing groups: (visit_num, date)
    segments = VISIT_HDR_RE.split(raw)
    # segments = [pre, num1, date1, content1, num2, date2, content2, ...]
    visits = []
    i = 1  # skip pre-match text at index 0
    while i + 2 < len(segments):
        date    = segments[i + 1].strip()
        content = segments[i + 2]
        # Remove long-dash dividers and strip
        content = DIVIDER_RE.sub("", content).strip()
        visits.append((date, content))
        i += 3
    return visits


def expand_row_to_visits(row: pd.Series) -> list:
    """
    Expand a (possibly multi-visit) CSV row into a list of single-visit dicts.
    Each dict has the same keys as the original row but with visit-specific values.
    """
    if not _is_multi_visit(row):
        # Single visit — return as-is (wrapped in list)
        visit = {col: _clean(row.get(col, "")) for col in row.index}
        # Parse the simple filename list
        raw_files = _clean(row.get(COL_TEN_FILE, ""))
        visit["_filenames"] = [
            f.strip() for f in raw_files.splitlines() if f.strip()
        ]
        visit["_date"] = _clean(row.get(COL_NGAY_DAU, ""))
        return [visit]

    # --- Multi-visit ---
    img_visits = parse_image_visits(_clean(row.get(COL_TEN_FILE, "")))
    n_visits   = len(img_visits)
    if n_visits == 0:
        return []

    # Parse each clinical column into per-visit sections
    clinical_visits: dict[str, list] = {}
    for col in CLINICAL_COLS:
        parsed = parse_clinical_visits(_clean(row.get(col, "")))
        # Pad or trim to match n_visits
        while len(parsed) < n_visits:
            parsed.append(("", ""))
        clinical_visits[col] = parsed[:n_visits]

    visits = []
    for idx in range(n_visits):
        date, filenames = img_visits[idx]
        v = {
            COL_MA_HO_SO: _clean(row.get(COL_MA_HO_SO, "")),
            COL_MA_BN:    _clean(row.get(COL_MA_BN, "")),
            COL_ICD:      _clean(row.get(COL_ICD, "")),
            "_date":      date,
            "_filenames": filenames,
        }
        for col in CLINICAL_COLS:
            _, content = clinical_visits[col][idx]
            v[col] = content
        visits.append(v)

    return visits


# ---------------------------------------------------------------------------
# Image index
# ---------------------------------------------------------------------------

def build_patient_image_index(patient_dir: Path) -> dict:
    """Recursively scan patient folder: {lowercase_filename → abs_path}."""
    index = {}
    for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
        for p in patient_dir.rglob(ext):
            index[p.name.lower()] = p
    return index


def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Clinical context + QA builders
# ---------------------------------------------------------------------------

def format_clinical_context(v: dict) -> str:
    parts = []
    if c := v.get(COL_LY_DO, ""):
        parts.append(f"Lý do vào viện: {c}.")
    if c := v.get(COL_QUA_TRINH, ""):
        parts.append(f"Quá trình bệnh lý: {c}.")
    if c := v.get(COL_TIEN_SU, ""):
        low = c.lower()
        if low not in ("chưa phát hiện bất thường", "chưa ghi nhận bất thường",
                       "chưa có", "không", "chưa"):
            parts.append(f"Tiền sử: {c}.")
    if c := v.get(COL_TOAN_THAN, ""):
        parts.append(f"Toàn thân: {c}.")
    if c := v.get(COL_CO_XUONG, ""):
        parts.append(f"Cơ xương khớp: {c}.")
    if c := v.get(COL_KHAM_BP, ""):
        parts.append(f"Khám bộ phận: {c}.")
    if c := v.get(COL_TOM_TAT, ""):
        if not _is_noise(c):
            parts.append(f"Tóm tắt CLS: {c}.")
    return " ".join(parts)


def make_diagnosis_answer(v: dict) -> str:
    kq   = v.get(COL_KQ_XN, "")
    chan = v.get(COL_CHAN_DOAN, "")
    if _is_noise(kq):
        kq = ""
    if kq and chan:
        if kq.lower() in chan.lower() or chan.lower() in kq.lower():
            return chan
        return f"Kết quả xét nghiệm: {kq}. Chẩn đoán xác định: {chan}."
    return chan or kq


def make_treatment_answer(v: dict) -> str:
    pp  = v.get(COL_PP_DT, "")
    hdt = v.get(COL_HUONG_DT, "")
    if _is_noise(pp):
        pp = ""
    if _is_noise(hdt):
        hdt = ""
    if pp and hdt:
        if pp.lower() == hdt.lower():
            return pp
        return f"Phương pháp điều trị: {pp}. Hướng điều trị tiếp theo: {hdt}."
    return pp or hdt


def make_qa_pairs(image_rel_path: str, visit: dict) -> list:
    context  = format_clinical_context(visit)
    ma_ho_so = visit.get(COL_MA_HO_SO, "")
    icd      = visit.get(COL_ICD, "")
    pairs    = []

    # Diagnosis
    dx_ans = make_diagnosis_answer(visit)
    if dx_ans:
        pairs.append({
            "image":    image_rel_path,
            "question": (
                f"Thông tin lâm sàng: {context} "
                f"Dựa vào hình ảnh X-quang và thông tin lâm sàng trên, "
                f"hãy đưa ra chẩn đoán xác định của bệnh nhân."
            ),
            "answer":   dx_ans,
            "qa_type":  "diagnosis",
            "ma_ho_so": ma_ho_so,
            "icd":      icd,
        })

    # Treatment (includes known diagnosis in question)
    tx_ans  = make_treatment_answer(visit)
    dx_known = visit.get(COL_CHAN_DOAN, "")
    if tx_ans:
        dx_suffix = f" Chẩn đoán xác định: {dx_known}." if dx_known else ""
        pairs.append({
            "image":    image_rel_path,
            "question": (
                f"Thông tin lâm sàng: {context}{dx_suffix} "
                f"Đề xuất phương pháp điều trị và hướng điều trị tiếp theo "
                f"phù hợp cho bệnh nhân này."
            ),
            "answer":   tx_ans,
            "qa_type":  "treatment",
            "ma_ho_so": ma_ho_so,
            "icd":      icd,
        })

    return pairs


# ---------------------------------------------------------------------------
# Split helpers
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
    parser.add_argument("--clinical_csv", default="./Ho_So_Lam_Sang_Day_Du.csv")
    parser.add_argument("--output_dir",   default="./data_enhanced")
    parser.add_argument("--train_ratio",  type=float, default=0.75)
    parser.add_argument("--val_ratio",    type=float, default=0.10)
    parser.add_argument("--test_ratio",   type=float, default=0.15)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6

    image_dir  = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load CSV
    # ------------------------------------------------------------------
    print("\n[1/4] Đọc CSV lâm sàng...")
    df = pd.read_csv(args.clinical_csv, dtype=str, encoding="utf-8-sig").fillna("")
    df.columns = [c.strip() for c in df.columns]
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    print(f"  Loaded {len(df)} rows từ CSV")

    # ------------------------------------------------------------------
    # 2. Expand multi-visit rows + patient split
    # ------------------------------------------------------------------
    print("\n[2/4] Expand multi-visit rows và tạo split...")

    all_pids = df[COL_MA_HO_SO].dropna().unique().tolist()
    train_pids, val_pids, test_pids = patient_split(
        all_pids, args.train_ratio, args.val_ratio, args.seed
    )
    print(f"  Bệnh nhân — train: {len(train_pids)} | val: {len(val_pids)} | test: {len(test_pids)}")

    stats  = defaultdict(int)
    splits = {"train": [], "val": [], "test": []}

    # ------------------------------------------------------------------
    # 3. Process each row → expand visits → find images → make QA pairs
    # ------------------------------------------------------------------
    print("\n[3/4] Tạo QA pairs (xử lý multi-visit)...")

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

        # Patient folder
        patient_dir = image_dir / ma_ho_so
        if not patient_dir.exists():
            stats["skip_no_folder"] += 1
            continue

        # Build image index once per patient folder
        img_index = build_patient_image_index(patient_dir)
        if not img_index:
            stats["skip_no_folder"] += 1
            continue

        # Expand to per-visit records
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

            # Check answers exist before touching images
            dx_ans = make_diagnosis_answer(visit)
            tx_ans = make_treatment_answer(visit)
            if not dx_ans and not tx_ans:
                stats["visit_no_answer"] += 1
                continue

            visit_has_img = False
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
                splits[split_name].extend(pairs)
                stats["total_qa"] += len(pairs)
                for p in pairs:
                    stats[f"qa_{p['qa_type']}"] += 1

                visit_has_img = True

            stats["visit_with_img" if visit_has_img else "visit_no_img"] += 1

    # ------------------------------------------------------------------
    # 4. Save JSONL
    # ------------------------------------------------------------------
    print("\n[4/4] Lưu JSONL...")
    for split_name, samples in splits.items():
        random.seed(args.seed)
        random.shuffle(samples)
        out_file = output_dir / f"dataset_{split_name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  {split_name:6s}: {len(samples):5d} QA pairs → {out_file}")

    # Stats
    stats_file = output_dir / "stats.txt"
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("=== Preprocess Enhanced Stats ===\n\n")
        f.write(f"CSV rows               : {len(df)}\n")
        f.write(f"Total visits expanded  : {stats['total_visits']}\n")
        f.write(f"Visits with images     : {stats['visit_with_img']}\n")
        f.write(f"Visits without images  : {stats['visit_no_img']}\n")
        f.write(f"Skipped (no patient ID): {stats['skip_no_id']}\n")
        f.write(f"Skipped (no folder)    : {stats['skip_no_folder']}\n")
        f.write(f"Skipped (no answer)    : {stats['visit_no_answer']}\n")
        f.write(f"Missing images         : {stats['missing_image']}\n")
        f.write(f"Invalid images         : {stats['invalid_image']}\n")
        f.write(f"Total QA pairs         : {stats['total_qa']}\n")
        f.write(f"  diagnosis            : {stats['qa_diagnosis']}\n")
        f.write(f"  treatment            : {stats['qa_treatment']}\n")
        f.write("\nSplit sizes:\n")
        for sn, sp in splits.items():
            f.write(f"  {sn}: {len(sp)}\n")

    print(f"\n  Stats → {stats_file}")
    print(f"\nDone! Tổng QA pairs: {stats['total_qa']}")
    print(f"  diagnosis : {stats['qa_diagnosis']}")
    print(f"  treatment : {stats['qa_treatment']}")
    print(f"  CSV rows expanded thành {stats['total_visits']} visits")


if __name__ == "__main__":
    main()
