"""
Modality ablation inference: full / text_only / image_only.

Extends the existing full-vs-minimal-context experiment (chapter 5, section
"Tổng kết chung") to the two extremes of removing an input modality entirely,
rather than just shrinking the clinical-text context:

  full        : image + full question text (clinical context + instruction).
                Same as run_inference.py's normal inference path.
  text_only   : full question text, images=None. No <image> token anywhere
                in the prompt (baseline/lora_ft use embed_tokens directly on
                text when images=None; xattn's cross-attn hooks already
                no-op when self.model._visual_tokens is None).
  image_only  : image only, generic qa_type-specific instruction
                ("hãy đưa ra chẩn đoán..." / "hãy đề xuất phương pháp điều
                trị..."), with all clinical-context text (Lý do vào viện,
                Tiền sử, Cơ xương khớp, ...) stripped out.

All three model types (baseline, lora_ft, xattn) support both ablations
mechanically without any architecture change — see prepare_inputs_labels_for_multimodal
in llava/model/llava_arch.py (images=None -> falls through to plain
embed_tokens) and LlavaMistralXAttnModel's cross-attn hooks (visual_tokens is
None -> hook returns layer_output unchanged).

Usage
-----
python run_inference_ablation.py \\
    --model_type xattn --modality text_only \\
    --model_path ./outputs_v2/stage2 --base_model microsoft/llava-med-v1.5-mistral-7b \\
    --data_path data_enhanced/dataset_test.jsonl --image_root ./bone_data \\
    --output_file results/ablation/xattn_s2_text_only.jsonl

python run_inference_ablation.py \\
    --model_type xattn --modality image_only \\
    --model_path ./outputs_v2/stage2 --base_model microsoft/llava-med-v1.5-mistral-7b \\
    --data_path data_enhanced/dataset_test.jsonl --image_root ./bone_data \\
    --output_file results/ablation/xattn_s2_image_only.jsonl
"""

import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from run_inference import load_baseline_model, load_lora_ft_model, load_xattn_model

# Generic, qa_type-specific instructions used for the image_only ablation —
# mirrors the instruction clauses in make_qa_pairs() in both preprocess_enhanced.py
# (diagnosis, treatment) and preprocess_test.py (+classification, xray_description),
# minus the "Thông tin lâm sàng: ..." context clause each of them prepends.
GENERIC_INSTRUCTIONS = {
    "diagnosis": "Dựa vào hình ảnh X-quang, hãy đưa ra chẩn đoán xác định của bệnh nhân.",
    "treatment": "Dựa vào hình ảnh X-quang, hãy đề xuất phương pháp điều trị và "
                 "hướng điều trị tiếp theo phù hợp cho bệnh nhân này.",
    "classification": "Dựa vào hình ảnh X-quang, bệnh nhân này thuộc nhóm bệnh xương nào "
                       "trong các nhóm sau: U xương, Nhiễm trùng xương, Thoái hóa xương, "
                       "Chấn thương xương, Chuyển hóa xương?",
    "xray_description": "Mô tả những bất thường thấy được trên hình ảnh X-quang này.",
}
DEFAULT_GENERIC_INSTRUCTION = GENERIC_INSTRUCTIONS["diagnosis"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_sample(line: dict) -> tuple:
    """Returns (question_id, image_filename, question_text, reference_answer, qa_type)."""
    if "text" in line:
        question = line["text"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
        reference = line.get("gpt4_answer", "")
    else:
        question = line.get("question", "")
        reference = line.get("answer", "")

    qid = line.get("question_id", line.get("id", 0))
    qa_type = line.get("qa_type", line.get("type", ""))
    return qid, line["image"], question, reference, qa_type


def build_text(question: str, qa_type: str, modality: str) -> str:
    """Text to place after (or instead of) the <image> token, per modality."""
    if modality in ("full", "text_only"):
        return question
    # image_only: drop clinical-context text, keep only a generic task instruction
    return GENERIC_INSTRUCTIONS.get(qa_type, DEFAULT_GENERIC_INSTRUCTION)


# ---------------------------------------------------------------------------
# Inference: baseline / lora_ft share the mm_projector prefix architecture
# ---------------------------------------------------------------------------

@torch.inference_mode()
def infer_prefix_style(
    model, tokenizer, image_processor,
    image_path: str, question: str, qa_type: str, modality: str,
    max_new_tokens: int, conv_mode: str = "mistral_instruct",
    prompt_style: str = "conv",
) -> str:
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images, KeywordsStoppingCriteria

    text = build_text(question, qa_type, modality)
    use_image = modality != "text_only"

    if prompt_style == "raw":
        # Mirrors infer_xattn()'s own completion-style convention exactly (no chat
        # template, no stop string) so baseline/lora_ft can be probed with the same
        # prompt XAttn was trained on -- isolates a prompt-template effect from an
        # architecture effect when the two model types are compared.
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{text}\nAnswer:" if use_image else f"{text}\nAnswer:"
        stop_str = None
    else:
        conv = conv_templates[conv_mode].copy()
        qs = (DEFAULT_IMAGE_TOKEN + "\n" + text) if use_image else text
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep2 if conv.sep2 else conv.sep

    model_dtype = next(model.parameters()).dtype

    if use_image:
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).to(dtype=model_dtype).cuda()
    else:
        # No <image> token anywhere in the prompt -> plain tokenizer, images=None.
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        image_tensor = None

    stopping_criteria = KeywordsStoppingCriteria(
        [stop_str] if stop_str else [], tokenizer, input_ids
    )

    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        stopping_criteria=[stopping_criteria] if stop_str else [],
    )
    # LlavaMistralForCausalLM.generate() always builds inputs_embeds internally
    # (both the images-not-None and images-None branches), so HF generate()
    # returns ONLY the newly generated tokens -- do NOT slice off
    # input_ids.shape[1] here (applies to use_image True AND False/text_only).
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if stop_str and answer.endswith(stop_str):
        answer = answer[: -len(stop_str)].strip()
    return answer


# ---------------------------------------------------------------------------
# Inference: XAttn (visual tokens injected via cross-attn hooks, not a prefix)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def infer_xattn(
    model, tokenizer, image_processor,
    image_path: str, question: str, qa_type: str, modality: str,
    max_new_tokens: int,
) -> str:
    text = build_text(question, qa_type, modality)
    use_image = modality != "text_only"

    # "<image>" here is only ever a literal text marker for xattn (never IMAGE_TOKEN_INDEX),
    # so it's dropped for text_only purely for narrative cleanliness, not correctness.
    prompt = f"{DEFAULT_IMAGE_TOKEN}\n{text}\nAnswer:" if use_image else f"{text}\nAnswer:"
    input_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).input_ids.cuda()

    if use_image:
        image = Image.open(image_path).convert("RGB")
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(dtype=torch.bfloat16).cuda()
    else:
        pixel_values = None  # forward()/generate() already no-op the cross-attn hooks on None

    output_ids = model.generate(
        inputs=input_ids,
        images=pixel_values,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if "Answer:" in decoded:
        decoded = decoded.split("Answer:")[-1].strip()
    return decoded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Modality ablation: full / text_only / image_only")
    parser.add_argument("--model_type", required=True, choices=["baseline", "xattn", "lora_ft"])
    parser.add_argument("--modality", required=True, choices=["full", "text_only", "image_only"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_model", default="microsoft/llava-med-v1.5-mistral-7b")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--conv_mode", default="mistral_instruct")
    parser.add_argument("--prompt_style", default="conv", choices=["conv", "raw"],
                        help="baseline/lora_ft only: 'conv' (default) uses the model's own "
                             "conv_templates chat format; 'raw' mirrors infer_xattn()'s "
                             "completion-style '...\\nAnswer:' prompt instead, to isolate a "
                             "prompt-template effect from an architecture effect when comparing "
                             "against xattn (e.g. the text_only modality gap). No-op for xattn.")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="xattn only, must match how the checkpoint was trained")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    print(f"\nLoading {args.model_type} model for modality='{args.modality}' "
          f"(prompt_style='{args.prompt_style}') ablation ...")
    if args.model_type == "baseline":
        tokenizer, model, image_processor = load_baseline_model(args.model_path)
        def infer_fn(img, q, t):
            return infer_prefix_style(model, tokenizer, image_processor, img, q, t, args.modality,
                                       args.max_new_tokens, args.conv_mode, args.prompt_style)
    elif args.model_type == "lora_ft":
        tokenizer, model, image_processor = load_lora_ft_model(args.model_path, args.base_model)
        def infer_fn(img, q, t):
            return infer_prefix_style(model, tokenizer, image_processor, img, q, t, args.modality,
                                       args.max_new_tokens, args.conv_mode, args.prompt_style)
    else:
        tokenizer, model, image_processor = load_xattn_model(
            args.model_path, args.base_model, load_in_8bit=args.load_in_8bit
        )
        def infer_fn(img, q, t):
            return infer_xattn(model, tokenizer, image_processor, img, q, t, args.modality,
                                args.max_new_tokens)

    with open(args.data_path, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(samples)} test samples from {args.data_path}")

    skipped, errors = 0, 0
    with open(args.output_file, "w", encoding="utf-8") as out_f:
        desc = f"Inferring ({args.model_type}, {args.modality})"
        for sample in tqdm(samples, desc=desc):
            qid, image_file, question, reference, qa_type = parse_sample(sample)
            image_path = os.path.join(args.image_root, image_file)

            # text_only never touches the image, so a missing file shouldn't skip the sample
            if args.modality != "text_only" and not os.path.exists(image_path):
                skipped += 1
                continue

            try:
                prediction = infer_fn(image_path, question, qa_type)
            except Exception as e:
                print(f"\n  [ERROR] qid={qid}: {e}")
                prediction = ""
                errors += 1

            record = {
                "question_id": qid,
                "image":       image_file,
                "question":    question,
                "reference":   reference,
                "prediction":  prediction,
                "model_type":  args.model_type,
                "modality":    args.modality,
                "qa_type":     qa_type,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    total = len(samples)
    print(f"\nDone. Processed {total - skipped - errors}/{total} samples.")
    if skipped:
        print(f"  Skipped {skipped} samples (image not found)")
    if errors:
        print(f"  Errors  {errors} samples (see messages above)")
    print(f"Results saved -> {args.output_file}")


if __name__ == "__main__":
    main()
