"""
Tạo test set với context tối giản để đánh giá visual reasoning thuần túy.

Thay vì full clinical context (có thể chứa hint chẩn đoán), chỉ giữ lại:
- Lý do vào viện (chief complaint)
- Câu hỏi gốc

Output: data_enhanced/dataset_test_minimal.jsonl
"""

import json
import re
import argparse


HINT_FIELDS = [
    "Quá trình bệnh lý",
    "Tóm tắt CLS",
    "Cận lâm sàng",
    "Chẩn đoán xác định",
    "Kết quả xét nghiệm",
]


def strip_hints(question: str) -> str:
    """Giữ lại Lý do vào viện + câu hỏi cuối, bỏ các field có hint."""
    lines = question.split(". ")

    # Tách phần context (trước câu hỏi cuối) và câu hỏi
    # Câu hỏi cuối thường là câu dài nhất không có dấu ":" hoặc bắt đầu bằng "Dựa vào"/"Đề xuất"
    question_starters = [
        "Dựa vào hình ảnh",
        "Đề xuất phương pháp",
        "Hãy đưa ra",
        "Mô tả những bất thường",
    ]

    # Tìm câu hỏi thực sự (phần cuối)
    actual_question = ""
    for starter in question_starters:
        if starter in question:
            idx = question.index(starter)
            actual_question = question[idx:].strip()
            question = question[:idx].strip()
            break

    # Từ context, chỉ giữ "Lý do vào viện"
    ly_do = ""
    match = re.search(r"Lý do vào viện:\s*([^.]+?)(?:\.|Quá trình|Tiền sử|Khám|Tóm tắt|$)", question)
    if match:
        ly_do = match.group(1).strip().rstrip(";").strip()

    # Xây dựng câu hỏi minimal
    minimal = "Thông tin lâm sàng:\n"
    if ly_do:
        minimal += f"Lý do vào viện: {ly_do}.\n"
    minimal += f"\n{actual_question}" if actual_question else ""

    return minimal.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data_enhanced/dataset_test.jsonl")
    parser.add_argument("--output", default="data_enhanced/dataset_test_minimal.jsonl")
    args = parser.parse_args()

    samples = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for s in samples:
            minimal_q = strip_hints(s["question"])
            if not minimal_q:
                continue
            record = dict(s)
            record["question"] = minimal_q
            record["original_question"] = s["question"]
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written {written} samples → {args.output}")

    # Preview 2 samples
    print("\n--- Preview ---")
    with open(args.output, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            s = json.loads(line)
            print(f"\n[{i+1}] Question:\n{s['question'][:300]}")
            print(f"     Answer: {s['answer'][:100]}")


if __name__ == "__main__":
    main()
