"""
Standalone re-verification for LoRA-FT — same monkey-patch technique as
run_baseline_textonly_fixed.py, does NOT modify run_inference.py or
llava_mistral.py. See GENERATE_BUG_INVESTIGATION.md for the full writeup.

load_lora_ft_model() calls PeftModel.from_pretrained(...).merge_and_unload(),
which returns a plain LlavaMistralForCausalLM instance -- so the same class-level
patch used for baseline applies directly here, no extra handling needed.

This tests the WITH-IMAGE branch of generate() (images is not None), which was
NOT covered by the earlier baseline text-only fix -- that only proved the
images-is-None branch was buggy. This script checks whether the images-is-not-None
branch (used for the officially reported LoRA-FT BERTScore-F1=0.051 at trang 29)
is also affected.

Usage:
    python run_lora_ft_fixed.py \
        --model_path ./outputs_lora_baseline_v3 --base_model microsoft/llava-med-v1.5-mistral-7b \
        --data_path bone_data/dataset_test.jsonl --image_root ./bone_data_mini \
        --output_file results/outputs/lora_baseline_v3_outputs_FIXED.jsonl
"""

import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm

from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM

# ---------------------------------------------------------------------------
# Same fix as run_baseline_textonly_fixed.py, but this time exercising the
# images-is-not-None branch too, since that's what infer_lora_ft() actually uses.
# ---------------------------------------------------------------------------
_original_generate = LlavaMistralForCausalLM.generate


def _fixed_generate(self, inputs=None, images=None, image_sizes=None, **kwargs):
    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if "inputs_embeds" in kwargs:
        raise NotImplementedError("`inputs_embeds` is not supported")

    if images is not None:
        (
            inputs, position_ids, attention_mask, _, inputs_embeds, _
        ) = self.prepare_inputs_labels_for_multimodal(
            inputs, position_ids, attention_mask, None, None, images, image_sizes=image_sizes
        )
        # FIX: prepare_inputs_labels_for_multimodal has already merged image+text
        # into inputs_embeds and returns a placeholder `inputs` (input_ids) too --
        # the original code discards this placeholder and passes inputs_embeds-only
        # to generate(). We instead keep inputs_embeds ONLY for step 0 (it must be
        # embeds here since it's the fused image+text sequence -- there's no valid
        # input_ids equivalent for the merged image tokens), but explicitly pass a
        # real attention_mask (already computed) so unfinished_sequences/cache
        # bookkeeping has a valid tensor to work from -- unlike the original,
        # which forwarded whatever attention_mask prepare_inputs_labels_for_multimodal
        # returned only through kwargs plumbing identically. The single actually-
        # different line vs. the original is dropped inputs_embeds path; keep it
        # since there's no input_ids alternative for fused multimodal embeds, but
        # add explicit attention_mask (already returned, non-None here) and route
        # through the same super() call used by the confirmed-fixed text-only path.
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long,
                                         device=inputs_embeds.device)
        return super(LlavaMistralForCausalLM, self).generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
    else:
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs)
        return super(LlavaMistralForCausalLM, self).generate(
            input_ids=inputs,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs,
        )


LlavaMistralForCausalLM.generate = _fixed_generate

from run_inference import load_lora_ft_model

# Same wording as run_inference_ablation.py's GENERIC_INSTRUCTIONS, reused here so
# the image_only ablation is directly comparable to the one already run for xattn.
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


def parse_sample(line):
    from llava.constants import DEFAULT_IMAGE_TOKEN
    if "text" in line:
        question = line["text"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
        reference = line.get("gpt4_answer", "")
    else:
        question = line.get("question", "")
        reference = line.get("answer", "")
    qid = line.get("question_id", line.get("id", 0))
    qa_type = line.get("qa_type", line.get("type", ""))
    return qid, line["image"], question, reference, qa_type


def build_text(question, qa_type, modality):
    """text to use for the given ablation modality (mirrors run_inference_ablation.py)."""
    if modality in ("full", "text_only"):
        return question
    return GENERIC_INSTRUCTIONS.get(qa_type, DEFAULT_GENERIC_INSTRUCTION)


def main():
    parser = argparse.ArgumentParser(description="Re-verify LoRA-FT (with-image) with the generate() fix")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_model", default="microsoft/llava-med-v1.5-mistral-7b")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--conv_mode", default="mistral_instruct")
    parser.add_argument("--modality", default="full", choices=["full", "text_only", "image_only"],
                        help="text_only drops the image entirely; image_only keeps the image but "
                             "replaces the clinical-context question with a generic qa_type "
                             "instruction. Together with full, these isolate whether the model is "
                             "actually using visual grounding or just reading clinical text.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    print("Loading LoRA-FT model (generate() patched, with-image branch) ...")
    tokenizer, model, image_processor = load_lora_ft_model(args.model_path, args.base_model)

    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

    @torch.inference_mode()
    def infer_lora_ft_fixed2(image_path, question, qa_type, max_new_tokens, conv_mode, modality="full"):
        # Reimplements infer_lora_ft() but fixes a second, independent bug found
        # after the generate() attention_mask fix: the original does
        # `generated_ids = output_ids[:, input_ids.shape[1]:]` assuming output_ids
        # includes the prompt -- true only when generate() is called with raw
        # input_ids. Here generate() necessarily uses inputs_embeds (image+text are
        # fused into a continuous embedding, no discrete input_ids equivalent
        # exists for the image tokens), so HF's returned sequence contains ONLY the
        # newly generated tokens already -- slicing by the (unrelated, shorter or
        # longer) pre-fusion prompt length truncates or empties real content.
        # Confirmed via direct A/B test: decoding output_ids[0] with no slicing
        # gives correct text; the original slice reliably empties it whenever the
        # answer is shorter than the prompt, which most clinical questions are.
        use_image = modality != "text_only"
        text = build_text(question, qa_type, modality)
        conv = conv_templates[conv_mode].copy()
        qs = (DEFAULT_IMAGE_TOKEN + "\n" + text) if use_image else text
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if use_image:
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).cuda()
            image = Image.open(image_path).convert("RGB")
            image_tensor = (
                image_processor(images=image, return_tensors="pt")
                .pixel_values.to(dtype=torch.bfloat16).cuda()
            )
        else:
            # No <image> token anywhere in the prompt -> plain tokenizer, images=None
            # -> _fixed_generate's images-is-None branch (raw input_ids, already fixed).
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            image_tensor = None
        stop_str = conv.sep2 if conv.sep2 else conv.sep
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str] if stop_str else [], tokenizer, input_ids
        )
        output_ids = model.generate(
            input_ids, images=image_tensor, do_sample=False,
            max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
            use_cache=True, stopping_criteria=[stopping_criteria] if stop_str else [],
        )
        if use_image:
            # FIX: no slicing by input_ids.shape[1] -- output_ids already contains
            # only the newly generated tokens when generate() ran on inputs_embeds
            # (image+text fused, no raw input_ids equivalent exists).
            answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        else:
            # text_only routes through _fixed_generate's images-is-None branch,
            # which passes raw input_ids -- HF's returned sequence DOES include
            # the prompt prefix here, so the original slicing is correct in this
            # one case only.
            generated_ids = output_ids[:, input_ids.shape[1]:]
            answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        if stop_str and answer.endswith(stop_str):
            answer = answer[: -len(stop_str)].strip()
        return answer

    with open(args.data_path, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(samples)} test samples from {args.data_path}")

    n_empty, n_skipped = 0, 0
    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for sample in tqdm(samples, desc="lora-ft-FIXED"):
            qid, image_file, question, reference, qa_type = parse_sample(sample)
            image_path = os.path.join(args.image_root, image_file)
            if args.modality != "text_only" and not os.path.exists(image_path):
                n_skipped += 1
                continue
            try:
                prediction = infer_lora_ft_fixed2(
                    image_path, question, qa_type, args.max_new_tokens, args.conv_mode, args.modality,
                )
            except Exception as e:
                print(f"\n  [ERROR] qid={qid}: {e}")
                prediction = ""
            if not prediction.strip():
                n_empty += 1
            record = {
                "question_id": qid, "image": image_file, "question": question,
                "reference": reference, "prediction": prediction,
                "model_type": "lora_ft",
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n_done = len(samples) - n_skipped
    print(f"\nDone. {n_empty}/{n_done} empty predictions "
          f"({100*n_empty/max(n_done,1):.1f}%), {n_skipped} skipped (missing image).")
    print(f"Written to {args.output_file}")


if __name__ == "__main__":
    main()
