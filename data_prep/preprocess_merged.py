"""
Preprocess Merged — reconstruction attempt of the official 8.798-sample
bone X-ray QA dataset, gộp logic từ 2 script gốc:

  - preprocess_enhanced.py : CSV parsing khớp đúng Ho_So_Lam_Sang_Day_Du.csv,
    multi-visit expansion (multi-visit rows dùng separator "── date ──" / "● [LẦN KHÁM n]")
  - ProcessData_fixed.ipynb: classify_image_type() (single/multi_view/multi_trauma,
    dùng để gộp TẤT CẢ ảnh của 1 visit vào 1 QA sample thay vì 1 QA/ảnh),
    logic 5-group (U_XUONG / NHIEM_TRUNG_XUONG / CHAN_THUONG_XUONG /
    THOAI_HOA_XUONG / XUONG_CHUYEN_HOA) + balancing.

QUAN TRỌNG — đây là bản TÁI DỰNG GẦN ĐÚNG, không phải script gốc:
  1. File nhãn nhóm bệnh gốc (mahs_theo_5_nhom_patient_mini.csv, do chuyên gia
     gán tay) KHÔNG có sẵn cho toàn bộ 12.014 hồ sơ → nhãn nhóm ở đây được suy
     ra TỰ ĐỘNG bằng từ khóa trên "Chẩn đoán xác định" + mã ICD-10 (heuristic),
     KHÔNG phải nhãn chuyên gia gốc. Dùng để ước lượng phân bố, không dùng để
     khẳng định số liệu chính thức.
  2. Bước cân bằng nhóm (Bước 7) dùng ngưỡng cap đơn giản (tham số --cap_per_group),
     không rõ thuật toán cân bằng gốc thực sự dùng.
  3. Kết quả CHẮC CHẮN sẽ không khớp tuyệt đối 8.798/6.807/704/1.287 — mục tiêu
     là tạo ra một bộ dữ liệu CÙNG QUY MÔ để tính các thống kê còn thiếu
     (phân bố nhóm bệnh, độ dài token câu hỏi/đáp án).

Usage:
  python preprocess_merged.py \\
      --image_dir ./bone_data --clinical_csv ./Ho_So_Lam_Sang_Day_Du.csv \\
      --output_dir ./data_merged --cap_per_group 1800
"""

import argparse
import json
import random
import re
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Column names (từ preprocess_enhanced.py, khớp đúng CSV thật)
# ---------------------------------------------------------------------------

COL_MA_HO_SO  = "Mã hồ sơ"
COL_MA_BN     = "Mã bệnh nhân"
COL_NGAY_DAU  = "Ngày khám (đầu)"
COL_TEN_FILE  = "Tên file .jpg (tất cả)"
COL_LY_DO     = "Lý do vào viện"
COL_QUA_TRINH = "Quá trình bệnh lý"
COL_TIEN_SU   = "Tiền sử bản thân"
COL_KHAM_BP   = "Khám các bộ phận"
COL_TOAN_THAN = "Khám toàn thân"
COL_CO_XUONG  = "Cơ xương khớp"
COL_TOM_TAT   = "Tóm tắt CLS"
COL_KQ_XN     = "Kết quả xét nghiệm"
COL_CHAN_DOAN = "Chẩn đoán xác định"
COL_PP_DT     = "Phương pháp điều trị"
COL_HUONG_DT  = "Hướng điều trị tiếp theo"
COL_ICD       = "Mã ICD-10"

CLINICAL_COLS = [
    COL_LY_DO, COL_QUA_TRINH, COL_TIEN_SU, COL_KHAM_BP,
    COL_TOAN_THAN, COL_CO_XUONG, COL_TOM_TAT,
    COL_KQ_XN, COL_CHAN_DOAN, COL_PP_DT, COL_HUONG_DT,
]

NOISE_TOKENS = {
    "xq", "xquang", "x-quang", "không", "khong", "chưa", "chua",
    "toa", "thuoc", "thuốc", "nội khoa", "ngoại khoa",
}

IMG_DATE_SEP_RE = re.compile(r'──\s*(\d{1,2}/\d{1,2}/\d{4})\s*──')
VISIT_HDR_RE = re.compile(
    r'●\s*\[LẦN KHÁM\s*(\d+)\s*\|\s*(\d{1,2}/\d{1,2}/\d{4})\s*\|[^\]]*\]'
)
DIVIDER_RE = re.compile(r'^─{10,}$', re.MULTILINE)


def _clean(val) -> str:
    if not isinstance(val, str):
        return ""
    v = val.strip()
    return "" if v.lower() in {"nan", "-", "—", ""} else v


def _is_noise(val: str) -> bool:
    v = val.strip().lower()
    return v in NOISE_TOKENS or len(v) <= 2


def _is_multi_visit(row: pd.Series) -> bool:
    img_raw = _clean(row.get(COL_TEN_FILE, ""))
    if IMG_DATE_SEP_RE.search(img_raw):
        return True
    for col in CLINICAL_COLS:
        if VISIT_HDR_RE.search(_clean(row.get(col, ""))):
            return True
    return False


def parse_image_visits(raw: str) -> list:
    if not raw.strip():
        return []
    parts = IMG_DATE_SEP_RE.split(raw)
    if len(parts) == 1:
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
    if not raw.strip():
        return [("", "")]
    if not VISIT_HDR_RE.search(raw):
        return [("", raw.strip())]
    segments = VISIT_HDR_RE.split(raw)
    visits = []
    i = 1
    while i + 2 < len(segments):
        date = segments[i + 1].strip()
        content = DIVIDER_RE.sub("", segments[i + 2]).strip()
        visits.append((date, content))
        i += 3
    return visits


def expand_row_to_visits(row: pd.Series) -> list:
    if not _is_multi_visit(row):
        visit = {col: _clean(row.get(col, "")) for col in row.index}
        raw_files = _clean(row.get(COL_TEN_FILE, ""))
        visit["_filenames"] = [f.strip() for f in raw_files.splitlines() if f.strip()]
        visit["_date"] = _clean(row.get(COL_NGAY_DAU, ""))
        return [visit]

    img_visits = parse_image_visits(_clean(row.get(COL_TEN_FILE, "")))
    n_visits = len(img_visits)
    if n_visits == 0:
        return []

    clinical_visits = {}
    for col in CLINICAL_COLS:
        parsed = parse_clinical_visits(_clean(row.get(col, "")))
        while len(parsed) < n_visits:
            parsed.append(("", ""))
        clinical_visits[col] = parsed[:n_visits]

    visits = []
    for idx in range(n_visits):
        date, filenames = img_visits[idx]
        v = {
            COL_MA_HO_SO: _clean(row.get(COL_MA_HO_SO, "")),
            COL_MA_BN: _clean(row.get(COL_MA_BN, "")),
            COL_ICD: _clean(row.get(COL_ICD, "")),
            "_date": date,
            "_filenames": filenames,
        }
        for col in CLINICAL_COLS:
            _, content = clinical_visits[col][idx]
            v[col] = content
        visits.append(v)
    return visits


def build_patient_image_index(patient_dir: Path) -> dict:
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
# classify_image_type (từ ProcessData_fixed.ipynb) — dùng để gộp ảnh 1 visit
# ---------------------------------------------------------------------------

ANATOMY_KW = [
    "đùi", "cẳng chân", "mác", "chày", "gối", "háng", "cổ chân",
    "cột sống", "vai", "khuỷu", "cẳng tay", "cổ tay", "bàn tay",
    "ngón tay", "xương sườn", "lồng ngực", "xương chậu",
    "cánh tay", "bàn chân", "ngón chân",
]


def classify_image_type(image_paths: list, clinical_text: str) -> str:
    if len(image_paths) <= 1:
        return "single"
    text_lower = str(clinical_text).lower()
    found = {kw for kw in ANATOMY_KW if kw in text_lower}
    return "multi_trauma" if len(found) >= 2 else "multi_view"


# ---------------------------------------------------------------------------
# Group classification — HEURISTIC THAY THẾ (không phải nhãn chuyên gia gốc)
# ---------------------------------------------------------------------------

GROUP_LABEL_VI = {
    "U_XUONG":           "U xương (Bone Tumor)",
    "NHIEM_TRUNG_XUONG": "Nhiễm trùng xương (Osteomyelitis)",
    "CHAN_THUONG_XUONG": "Gãy xương / Chấn thương cơ xương (Fracture/Trauma)",
    "THOAI_HOA_XUONG":   "Bệnh xương thoái hóa (Degenerative Bone Disease)",
    "XUONG_CHUYEN_HOA":  "Bệnh xương chuyển hóa (Metabolic Bone Disease)",
}

# Từ khóa suy ra nhóm từ "Chẩn đoán xác định" — thứ tự ưu tiên quan trọng
GROUP_KEYWORDS = [
    ("U_XUONG", ["u xương", "u lành", "u ác", "sarcoma", "u nang xương",
                 "khối u", "tumor", "carcinoma", "di căn xương"]),
    ("NHIEM_TRUNG_XUONG", ["viêm xương", "viêm tủy xương", "nhiễm trùng xương",
                            "osteomyelitis", "áp xe xương", "lao xương"]),
    ("THOAI_HOA_XUONG", ["thoái hóa", "loãng xương", "osteoporosis",
                          "thoát vị đĩa đệm", "hẹp ống sống", "gai cột sống"]),
    ("XUONG_CHUYEN_HOA", ["chuyển hóa", "còi xương", "nhuyễn xương",
                           "cường cận giáp", "paget"]),
    ("CHAN_THUONG_XUONG", ["gãy", "trật khớp", "chấn thương", "vỡ xương",
                            "gãy kín", "gãy hở", "đa chấn thương"]),
]

# ICD-10 prefix fallback (khi không khớp từ khóa văn bản)
ICD_PREFIX_GROUP = [
    ("C4", "U_XUONG"), ("D48", "U_XUONG"), ("D16", "U_XUONG"),
    ("M86", "NHIEM_TRUNG_XUONG"), ("M46", "NHIEM_TRUNG_XUONG"),
    ("M80", "THOAI_HOA_XUONG"), ("M81", "THOAI_HOA_XUONG"),
    ("M15", "THOAI_HOA_XUONG"), ("M19", "THOAI_HOA_XUONG"),
    ("M47", "THOAI_HOA_XUONG"), ("M51", "THOAI_HOA_XUONG"),
    ("E83", "XUONG_CHUYEN_HOA"), ("E55", "XUONG_CHUYEN_HOA"),
    ("S", "CHAN_THUONG_XUONG"), ("T", "CHAN_THUONG_XUONG"),
    ("M84", "CHAN_THUONG_XUONG"),
]


def classify_group(chan_doan: str, icd: str) -> str:
    text = chan_doan.lower()
    for group, kws in GROUP_KEYWORDS:
        if any(kw in text for kw in kws):
            return group
    icd_upper = icd.upper().strip()
    for prefix, group in ICD_PREFIX_GROUP:
        if icd_upper.startswith(prefix):
            return group
    return "CHAN_THUONG_XUONG"  # fallback: nhóm phổ biến nhất trong X-quang xương


# ---------------------------------------------------------------------------
# Clinical context + QA builders (per-visit, không explode theo ảnh)
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


def format_minimal_context(v: dict) -> str:
    """Minimal Context (Mục 3.4.1, chapter3): chỉ ảnh X-quang + lý do vào viện,
    không kèm quá trình bệnh lý / tiền sử / khám / Tóm tắt CLS."""
    c = v.get(COL_LY_DO, "")
    return f"Lý do vào viện: {c}." if c else ""


def make_diagnosis_answer(v: dict) -> str:
    kq = v.get(COL_KQ_XN, "")
    chan = v.get(COL_CHAN_DOAN, "")
    if _is_noise(kq):
        kq = ""
    if kq and chan:
        if kq.lower() in chan.lower() or chan.lower() in kq.lower():
            return chan
        return f"Kết quả xét nghiệm: {kq}. Chẩn đoán xác định: {chan}."
    return chan or kq


def make_treatment_answer(v: dict) -> str:
    pp = v.get(COL_PP_DT, "")
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


def make_qa_pairs_for_visit(image_rel_paths: list, visit: dict, img_note: str,
                             minimal_context: bool = False) -> list:
    context = format_minimal_context(visit) if minimal_context else format_clinical_context(visit)
    ma_ho_so = visit.get(COL_MA_HO_SO, "")
    icd = visit.get(COL_ICD, "")
    chan_doan = visit.get(COL_CHAN_DOAN, "")
    group = classify_group(chan_doan, icd)

    pairs = []

    dx_ans = make_diagnosis_answer(visit)
    if dx_ans:
        pairs.append({
            "image": image_rel_paths[0],  # train_xattn.py/train_lora_baseline_v3.py expect one image path per sample
            "question": (
                f"{img_note} Thông tin lâm sàng: {context} "
                f"Dựa vào hình ảnh X-quang và thông tin lâm sàng trên, "
                f"hãy đưa ra chẩn đoán xác định của bệnh nhân."
            ),
            "answer": dx_ans,
            "qa_type": "diagnosis",
            "ma_ho_so": ma_ho_so,
            "icd": icd,
            "group": group,
        })

    tx_ans = make_treatment_answer(visit)
    if tx_ans:
        dx_suffix = f" Chẩn đoán xác định: {chan_doan}." if chan_doan else ""
        pairs.append({
            "image": image_rel_paths[0],  # train_xattn.py/train_lora_baseline_v3.py expect one image path per sample
            "question": (
                f"{img_note} Thông tin lâm sàng: {context}{dx_suffix} "
                f"Đề xuất phương pháp điều trị và hướng điều trị tiếp theo "
                f"phù hợp cho bệnh nhân này."
            ),
            "answer": tx_ans,
            "qa_type": "treatment",
            "ma_ho_so": ma_ho_so,
            "icd": icd,
            "group": group,
        })

    return pairs


def patient_split(pids: list, train_r: float, val_r: float, seed: int):
    random.seed(seed)
    pids = list(pids)
    random.shuffle(pids)
    n = len(pids)
    n_tr = int(n * train_r)
    n_val = int(n * val_r)
    return set(pids[:n_tr]), set(pids[n_tr:n_tr + n_val]), set(pids[n_tr + n_val:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="./bone_data")
    parser.add_argument("--clinical_csv", default="./Ho_So_Lam_Sang_Day_Du.csv")
    parser.add_argument("--output_dir", default="./data_merged")
    parser.add_argument("--train_ratio", type=float, default=0.75)
    parser.add_argument("--val_ratio", type=float, default=0.08)
    parser.add_argument("--test_ratio", type=float, default=0.17)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap_per_group", type=int, default=0,
                         help="Nếu >0: giới hạn tối đa mỗi (group, qa_type) còn "
                              "bấy nhiêu mẫu trong TRAIN (Bước 7 - cân bằng, xấp xỉ).")
    parser.add_argument("--minimal_context", action="store_true",
                         help="Dùng Minimal Context (chỉ ảnh + lý do vào viện) thay vì "
                              "Full Context, để so được với số liệu minimal-context chính "
                              "của Chương 4 (0,678/0,726 LGCA; 0,525/0,529 LoRA r=16).")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Đọc CSV lâm sàng...")
    df = pd.read_csv(args.clinical_csv, dtype=str, encoding="utf-8-sig").fillna("")
    df.columns = [c.strip() for c in df.columns]
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    print(f"  Loaded {len(df)} rows từ CSV")

    print("\n[2/4] Expand multi-visit rows và tạo split theo bệnh nhân...")
    all_pids = df[COL_MA_HO_SO].dropna().unique().tolist()
    train_pids, val_pids, test_pids = patient_split(
        all_pids, args.train_ratio, args.val_ratio, args.seed
    )
    print(f"  Bệnh nhân — train: {len(train_pids)} | val: {len(val_pids)} | test: {len(test_pids)}")

    stats = defaultdict(int)
    splits = {"train": [], "val": [], "test": []}

    print("\n[3/4] Tạo QA pairs theo VISIT (gộp toàn bộ ảnh của 1 visit vào 1 sample)...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Rows"):
        ma_ho_so = _clean(str(row.get(COL_MA_HO_SO, "")))
        if not ma_ho_so:
            stats["skip_no_id"] += 1
            continue

        split_name = (
            "train" if ma_ho_so in train_pids else
            "val" if ma_ho_so in val_pids else
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

            valid_rel_paths = []
            for filename in filenames:
                abs_path = img_index.get(filename.lower())
                if abs_path is None:
                    stats["missing_image"] += 1
                    continue
                if not is_valid_image(str(abs_path)):
                    stats["invalid_image"] += 1
                    continue
                # .as_posix() (not str()) so the path always uses forward slashes,
                # even when this script runs on Windows — the JSONL is consumed by
                # training code that may run on Linux (e.g. a Kaggle kernel), where
                # a literal backslash is just a filename character, not a separator.
                valid_rel_paths.append(abs_path.relative_to(image_dir).as_posix())

            if not valid_rel_paths:
                stats["visit_no_img"] += 1
                continue
            stats["visit_with_img"] += 1

            clinical_ctx = format_clinical_context(visit)
            img_type = classify_image_type(valid_rel_paths, clinical_ctx)
            if img_type == "multi_trauma":
                img_note = (f"Bệnh nhân có {len(valid_rel_paths)} ảnh X-quang tương ứng "
                            "với các vùng chấn thương khác nhau.")
            elif img_type == "multi_view":
                img_note = f"Có {len(valid_rel_paths)} góc chụp X-quang của cùng vùng tổn thương."
            else:
                img_note = "Ảnh X-quang vùng tổn thương."

            pairs = make_qa_pairs_for_visit(valid_rel_paths, visit, img_note,
                                             minimal_context=args.minimal_context)
            splits[split_name].extend(pairs)
            stats["total_qa"] += len(pairs)
            for p in pairs:
                stats[f"qa_{p['qa_type']}"] += 1
                stats[f"group_{p['group']}"] += 1

    # ------------------------------------------------------------------
    # Bước 7 (xấp xỉ): cân bằng nhóm — chỉ áp dụng trên TRAIN nếu có --cap_per_group
    # ------------------------------------------------------------------
    if args.cap_per_group > 0:
        print(f"\n[3b] Cân bằng nhóm trên TRAIN (cap={args.cap_per_group} mỗi group x qa_type)...")
        by_key = defaultdict(list)
        for s in splits["train"]:
            by_key[(s["group"], s["qa_type"])].append(s)
        random.seed(args.seed)
        capped = []
        for key, items in by_key.items():
            random.shuffle(items)
            capped.extend(items[:args.cap_per_group])
        splits["train"] = capped

    print("\n[4/4] Lưu JSONL...")
    for split_name, samples in splits.items():
        random.seed(args.seed)
        random.shuffle(samples)
        out_file = output_dir / f"dataset_{split_name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  {split_name:6s}: {len(samples):5d} QA pairs -> {out_file}")

    total = sum(len(v) for v in splits.values())
    print(f"\nTong QA pairs (sau {'cap' if args.cap_per_group else 'khong cap'}): {total}")
    print(f"Muc tieu chinh thuc: 8798 (train 6807 / val 704 / test 1287)")

    print("\nPhan bo nhom benh (uoc luong tu heuristic ICD/tu khoa, TRAIN):")
    train_group_counts = Counter(s["group"] for s in splits["train"])
    for g, c in sorted(train_group_counts.items(), key=lambda x: -x[1]):
        pct = 100 * c / max(len(splits["train"]), 1)
        print(f"  {g:<20} {c:>6}  ({pct:.1f}%)  {GROUP_LABEL_VI.get(g, g)}")

    stats_file = output_dir / "stats.txt"
    with open(stats_file, "w", encoding="utf-8") as f:
        for k, v in sorted(stats.items()):
            f.write(f"{k}: {v}\n")
        f.write(f"\ntotal_after_output: {total}\n")


if __name__ == "__main__":
    main()
