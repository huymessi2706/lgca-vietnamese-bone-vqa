"""
augment_minority.py
===================
Tạo thêm QA pairs cho các nhóm thiểu số để cân bằng dataset.

Chiến lược:
  NHIEM_TRUNG (131 BN)       × 8 QA types = ~1,048
  XUONG_CHUYEN_HOA (347 BN)  × 5 QA types = ~1,735
  U_XUONG (886 BN)           × 4 QA types = ~3,544
  THOAI_HOA (cap 1,000 BN)   × 3 QA types = ~3,000
  CHAN_THUONG (cap 1,000 BN)  × 3 QA types = ~3,000
  ─────────────────────────────────────────────────
  Tổng ≈ 12,327 samples

Chạy SAU khi đã có output_dataset/instruction_tuning.jsonl từ ProcessData_fixed
"""

import json
import os
import random
from collections import defaultdict

# ── Đường dẫn ──────────────────────────────────────────────
INPUT_JSONL  = "./output_dataset/instruction_tuning.jsonl"
OUTPUT_JSONL = "./output_dataset/instruction_tuning_balanced.jsonl"
RANDOM_SEED  = 42

# Số QA pairs tối đa mỗi nhóm muốn đạt được
TARGET_SAMPLES = {
    "NHIEM_TRUNG_XUONG": 1000,
    "XUONG_CHUYEN_HOA":  1500,
    "U_XUONG":           3000,
    "THOAI_HOA_XUONG":   3000,
    "CHAN_THUONG_XUONG":  3000,
}

SYSTEM_PROMPT = (
    "Bạn là hệ thống trợ lý chẩn đoán y khoa chuyên về bệnh lý cơ-xương-khớp. "
    "Phân tích ảnh X-quang và thông tin lâm sàng để đưa ra nhận định chẩn đoán "
    "chính xác và có căn cứ lâm sàng."
)

SKIP_VALUES = {"", "không", "nan", "none", "chưa ghi nhận bất thường",
               "chưa phát hiện bất thường", "n/a", "na"}

GROUP_LABEL_VI = {
    "U_XUONG":           "U xương (Bone Tumor)",
    "NHIEM_TRUNG_XUONG": "Nhiễm trùng xương (Osteomyelitis)",
    "CHAN_THUONG_XUONG":  "Gãy xương / Chấn thương (Fracture/Trauma)",
    "THOAI_HOA_XUONG":   "Bệnh xương thoái hóa (Degenerative)",
    "XUONG_CHUYEN_HOA":  "Bệnh xương chuyển hóa (Metabolic)",
}


# ── Helper ──────────────────────────────────────────────────
def _val(sample: dict, key: str) -> str:
    v = str(sample.get(key, "")).strip()
    return "" if v.lower() in SKIP_VALUES else v


def _img_note(sample: dict) -> str:
    t = sample.get("image_type", "single")
    n = len(sample.get("image", []))
    if t == "multi_trauma":
        return f"Bệnh nhân có {n} ảnh X-quang tương ứng với các vùng chấn thương khác nhau."
    if t == "multi_view":
        return f"Có {n} góc chụp X-quang của cùng vùng tổn thương."
    return "Ảnh X-quang vùng tổn thương."


# ── Extra QA generators ──────────────────────────────────────

def qa_differential(sample: dict) -> dict | None:
    """So sánh/phân biệt với nhóm bệnh khác."""
    group    = sample.get("group", "")
    ctx      = sample.get("clinical_context", "")
    img_note = _img_note(sample)
    group_vi = GROUP_LABEL_VI.get(group, group)

    DIFF_HINTS = {
        "NHIEM_TRUNG_XUONG": (
            "gãy xương thông thường hoặc u xương",
            "Nhiễm trùng xương thường biểu hiện: sốt, bạch cầu tăng, ESR/CRP tăng cao, "
            "trên X-quang thấy phá hủy xương, phản ứng màng xương (periosteal reaction), "
            "có thể kèm áp xe phần mềm — khác với gãy xương (tiền sử chấn thương rõ) "
            "và u xương (phát triển chậm, không sốt)."
        ),
        "U_XUONG": (
            "nhiễm trùng xương hoặc chấn thương",
            "U xương thường không có tiền sử chấn thương rõ ràng, phát triển từ từ, "
            "trên X-quang thấy tổn thương dạng khối, có thể có vôi hóa, phá hủy vỏ xương "
            "hoặc tam giác Codman (sarcom). Nhiễm trùng có sốt + CRP tăng. "
            "Chấn thương có tiền sử tai nạn rõ ràng."
        ),
        "XUONG_CHUYEN_HOA": (
            "loãng xương so với nguyên nhân khác",
            "Bệnh xương chuyển hóa (loãng xương, còi xương) thường ảnh hưởng toàn thân: "
            "mật độ xương giảm lan tỏa, dễ gãy do lực nhẹ (gãy bệnh lý), "
            "xét nghiệm: Ca, P, PTH, Vitamin D bất thường. "
            "Khác với u xương (tổn thương khu trú) và nhiễm trùng (sốt, CRP tăng)."
        ),
        "THOAI_HOA_XUONG": (
            "thoái hóa so với chấn thương cấp tính",
            "Bệnh thoái hóa xương-khớp: triệu chứng từ từ, đau tăng khi vận động, "
            "X-quang: hẹp khe khớp, gai xương (osteophyte), đặc xương dưới sụn. "
            "Khác với chấn thương cấp (đau đột ngột sau tai nạn, phù nề cấp tính)."
        ),
        "CHAN_THUONG_XUONG": (
            "gãy xương do chấn thương so với gãy bệnh lý",
            "Gãy do chấn thương: tiền sử tai nạn/té ngã rõ ràng, đường gãy thường đơn giản, "
            "không có tổn thương nền. Gãy bệnh lý: xảy ra với lực nhỏ, "
            "X-quang thấy tổn thương nền (u xương, loãng xương) quanh ổ gãy."
        ),
    }

    if group not in DIFF_HINTS:
        return None

    compare_with, answer = DIFF_HINTS[group]
    q = (
        f"<image>\n{img_note}\n\n"
        f"Thông tin lâm sàng:\n{ctx}\n\n"
        f"Hãy phân biệt bệnh lý này ({group_vi}) với {compare_with}. "
        "Nêu các đặc điểm lâm sàng và hình ảnh X-quang giúp phân biệt."
    )
    return {"qa_type": "differential", "question": q, "answer": answer}


def qa_severity(sample: dict) -> dict | None:
    """Đánh giá mức độ nặng — hữu ích cho NHIEM_TRUNG, U_XUONG."""
    group    = sample.get("group", "")
    ctx      = sample.get("clinical_context", "")
    xq       = _val(sample, "xq_result") or ctx[:200]
    img_note = _img_note(sample)

    SEVERITY_TEMPLATE = {
        "NHIEM_TRUNG_XUONG": (
            "Đánh giá mức độ nhiễm trùng xương của bệnh nhân này. "
            "Phân loại: nhẹ (chưa lan rộng), trung bình (có phá hủy xương khu trú), "
            "nặng (phá hủy lan rộng, áp xe, hoại tử). Giải thích dựa trên lâm sàng và X-quang.",
            f"Dựa trên lâm sàng: {ctx[:300]}\nKết quả hình ảnh: {xq}\n"
            "→ Cần đánh giá: mức độ phá hủy xương, lan rộng phần mềm, đáp ứng điều trị."
        ),
        "U_XUONG": (
            "Dựa vào ảnh X-quang và thông tin lâm sàng, đánh giá giai đoạn/mức độ "
            "của khối u xương này (lành tính/ác tính nghi ngờ, khu trú/lan rộng).",
            f"Dựa trên: {ctx[:300]}\nHình ảnh: {xq}\n"
            "→ Đặc điểm cần chú ý: ranh giới khối u, phá hủy vỏ xương, "
            "phản ứng màng xương, xâm lấn mô mềm."
        ),
    }

    if group not in SEVERITY_TEMPLATE:
        return None

    q_text, answer = SEVERITY_TEMPLATE[group]
    q = f"<image>\n{img_note}\n\nThông tin lâm sàng:\n{ctx}\n\n{q_text}"
    return {"qa_type": "severity", "question": q, "answer": answer}


def qa_treatment_rationale(sample: dict) -> dict | None:
    """Giải thích lý do chọn phương pháp điều trị."""
    ctx      = sample.get("clinical_context", "")
    dieu_tri = _val(sample, "dieu_tri") or "phẫu thuật chỉnh hình phù hợp"
    group    = sample.get("group", "")
    img_note = _img_note(sample)

    if not dieu_tri or len(dieu_tri) < 10:
        return None

    q = (
        f"<image>\n{img_note}\n\n"
        f"Thông tin lâm sàng:\n{ctx}\n\n"
        f"Phương pháp điều trị đã chọn: {dieu_tri}\n\n"
        "Giải thích tại sao phương pháp điều trị này phù hợp với tình trạng bệnh nhân."
    )
    answer = (
        f"Phương pháp '{dieu_tri}' được chọn dựa trên:\n"
        f"• Chẩn đoán xác định thuộc nhóm {GROUP_LABEL_VI.get(group, group)}\n"
        f"• Tình trạng lâm sàng: {ctx[:200]}\n"
        "• Mục tiêu: kiểm soát bệnh lý, phục hồi chức năng, hạn chế biến chứng."
    )
    return {"qa_type": "treatment_rationale", "question": q, "answer": answer}


def qa_xray_finding(sample: dict) -> dict | None:
    """Tập trung mô tả phát hiện trên X-quang."""
    xq       = _val(sample, "xq_result")
    img_note = _img_note(sample)
    group    = sample.get("group", "")

    if not xq or len(xq) < 15:
        return None

    q = (
        f"<image>\n{img_note}\n\n"
        "Mô tả chi tiết các phát hiện bất thường trên ảnh X-quang này. "
        "Tập trung vào vị trí, hình thái, mức độ tổn thương xương."
    )
    return {"qa_type": "xray_finding", "question": q, "answer": xq}


def qa_followup(sample: dict) -> dict | None:
    """Hướng theo dõi sau điều trị."""
    huong    = _val(sample, "huong_tiep")
    ctx      = sample.get("clinical_context", "")
    img_note = _img_note(sample)

    if not huong or len(huong) < 10:
        return None

    q = (
        f"<image>\n{img_note}\n\n"
        f"Thông tin lâm sàng:\n{ctx}\n\n"
        "Đề xuất kế hoạch theo dõi và tái khám cho bệnh nhân này sau điều trị."
    )
    return {"qa_type": "followup", "question": q, "answer": huong}


def qa_clinical_reasoning(sample: dict) -> dict | None:
    """Lý luận lâm sàng — tại sao chẩn đoán này."""
    group    = sample.get("group", "")
    ctx      = sample.get("clinical_context", "")
    chan_doan = _val(sample, "chan_doan") or GROUP_LABEL_VI.get(group, group)
    img_note = _img_note(sample)

    if not ctx or len(ctx) < 30:
        return None

    q = (
        f"<image>\n{img_note}\n\n"
        f"Thông tin lâm sàng:\n{ctx}\n\n"
        "Trình bày lý luận lâm sàng dẫn đến chẩn đoán xác định. "
        "Nêu các dấu hiệu then chốt ủng hộ chẩn đoán này."
    )
    answer = (
        f"Chẩn đoán xác định: **{chan_doan}**\n\n"
        f"Lý luận lâm sàng:\n{ctx[:400]}\n\n"
        f"Nhóm bệnh lý: {GROUP_LABEL_VI.get(group, group)}"
    )
    return {"qa_type": "clinical_reasoning", "question": q, "answer": answer}


# Extra QA types theo nhóm
EXTRA_QA_GENERATORS = {
    "NHIEM_TRUNG_XUONG": [
        qa_differential, qa_severity, qa_treatment_rationale,
        qa_xray_finding, qa_followup,
    ],
    "XUONG_CHUYEN_HOA": [
        qa_differential, qa_treatment_rationale, qa_xray_finding,
    ],
    "U_XUONG": [
        qa_differential, qa_severity, qa_xray_finding,
    ],
    "THOAI_HOA_XUONG": [
        qa_differential, qa_xray_finding,
    ],
    "CHAN_THUONG_XUONG": [],
}


def generate_extra_pairs(sample: dict) -> list:
    """Sinh thêm QA pairs từ một sample gốc."""
    group = sample.get("group", "")
    extra = []

    # Chuẩn bị các trường hay dùng
    enriched = dict(sample)
    enriched["xq_result"]  = _val(sample, "xq_result")  if "xq_result"  in sample else ""
    enriched["chan_doan"]   = _val(sample, "chan_doan")   if "chan_doan"  in sample else ""
    enriched["dieu_tri"]    = _val(sample, "dieu_tri")    if "dieu_tri"   in sample else ""
    enriched["huong_tiep"]  = _val(sample, "huong_tiep")  if "huong_tiep" in sample else ""

    for gen_fn in EXTRA_QA_GENERATORS.get(group, []):
        result = gen_fn(enriched)
        if result is None:
            continue
        new_sample = {
            "id":               sample["id"],
            "image":            sample["image"],
            "image_type":       sample.get("image_type", "single"),
            "system":           SYSTEM_PROMPT,
            "clinical_context": sample.get("clinical_context", ""),
            "group":            group,
            "icd":              sample.get("icd", ""),
            "qa_type":          result["qa_type"],
            "conversations": [
                {"role": "user",      "content": result["question"]},
                {"role": "assistant", "content": result["answer"]},
            ],
        }
        extra.append(new_sample)

    return extra


# ── MAIN ─────────────────────────────────────────────────────
def run():
    random.seed(RANDOM_SEED)

    print("=" * 65)
    print("AUGMENTATION — Cân bằng dataset theo 5 nhóm bệnh xương")
    print("=" * 65)

    # 1. Load
    print(f"\n[1/3] Load: {INPUT_JSONL}")
    with open(INPUT_JSONL, encoding="utf-8") as f:
        all_samples = [json.loads(l) for l in f]
    print(f"   Tổng samples gốc: {len(all_samples):,}")

    # Group samples by patient id (mỗi BN có 3 qa types)
    by_patient: dict[str, list] = defaultdict(list)
    for s in all_samples:
        by_patient[s["id"]].append(s)

    by_group: dict[str, list] = defaultdict(list)
    for pid, samples in by_patient.items():
        grp = samples[0]["group"]
        by_group[grp].append(pid)

    print("\n  Phân bổ ban đầu (BN):")
    for g, ids in sorted(by_group.items(), key=lambda x: -len(x[1])):
        print(f"    {g:<25}: {len(ids):>5} BN → {len(ids)*3:>6} samples")

    # 2. Augment + sample
    print("\n[2/3] Augment & balance...")
    final_samples = []
    stats = {}

    for group, target in TARGET_SAMPLES.items():
        patient_ids = by_group.get(group, [])
        if not patient_ids:
            print(f"   ⚠ Không có BN nào cho nhóm {group}")
            continue

        # Với nhóm lớn hơn target/3 BN → undersample trước
        max_patients = target // 3 + 1
        if len(patient_ids) > max_patients:
            selected_ids = random.sample(patient_ids, max_patients)
        else:
            selected_ids = patient_ids

        selected_set = set(selected_ids)
        base = [s for pid in selected_set for s in by_patient[pid]]

        # Sinh extra QA
        augmented = []
        if EXTRA_QA_GENERATORS.get(group):
            for pid in selected_ids:
                # Lấy 1 sample đại diện (closed) để sinh extra
                rep = next((s for s in by_patient[pid]
                            if s.get("qa_type") == "closed"), by_patient[pid][0])
                augmented.extend(generate_extra_pairs(rep))

        combined = base + augmented

        # Nếu vẫn ít hơn target → oversample (lặp lại augmented)
        if len(combined) < target and augmented:
            deficit = target - len(combined)
            combined += random.choices(augmented, k=deficit)

        # Nếu nhiều hơn target → cắt bớt
        if len(combined) > target:
            combined = random.sample(combined, target)

        final_samples.extend(combined)
        stats[group] = {
            "patients": len(selected_ids),
            "base": len(base),
            "augmented": len(augmented),
            "final": len(combined),
        }
        print(f"   {group:<25}: {len(selected_ids):>4} BN | "
              f"base={len(base):>5} + aug={len(augmented):>4} → final={len(combined):>5}")

    # Shuffle
    random.shuffle(final_samples)

    # 3. Ghi output
    print(f"\n[3/3] Ghi: {OUTPUT_JSONL}")
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for s in final_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("\n" + "=" * 65)
    print("KẾT QUẢ SAU AUGMENTATION")
    print("=" * 65)
    total = sum(s["final"] for s in stats.values())
    for g in ["U_XUONG", "NHIEM_TRUNG_XUONG", "XUONG_CHUYEN_HOA", "THOAI_HOA_XUONG", "CHAN_THUONG_XUONG"]:
        if g in stats:
            s = stats[g]
            bar = "█" * (s["final"] // 100)
            print(f"  {g:<25}: {s['final']:>5}  {bar}")
    print(f"\n  Tổng samples balanced : {total:,}")
    print(f"  Output: {OUTPUT_JSONL}")
    print("=" * 65)


if __name__ == "__main__":
    run()
