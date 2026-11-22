"""
Standalone re-verification script — does NOT modify run_inference_ablation.py or
llava_mistral.py. Monkey-patches the confirmed generate() bug at runtime only
within this process, then reuses the existing (untouched) infer_prefix_style()
and load_baseline_model() to regenerate baseline text_only predictions.

Root cause (confirmed via 10/10 reproducible trials on 2026-08-05): when
`images is None`, LlavaMistralForCausalLM.generate() computes inputs_embeds
itself and passes ONLY inputs_embeds (no input_ids) to GenerationMixin.generate().
Under transformers==4.36.2 this call is non-deterministic per-process: sometimes
returns 0 generated tokens despite step-0 logits being unambiguous (verified via
direct logits comparison -- baseline and xattn produce bit-identical logits given
identical weights/prompt, yet the multi-step loop can silently emit nothing).
Passing input_ids directly instead (letting forward()'s own embed_tokens
fallback run per-step, same as it already does for xattn) fixed this 5/5 times
in isolated testing.

Usage:
    python run_baseline_textonly_fixed.py --prompt_style conv \
        --data_path bone_data/dataset_test.jsonl --image_root ./bone_data_mini \
        --output_file results/ablation/baseline_text_only_FIXED.jsonl

    python run_baseline_textonly_fixed.py --prompt_style raw \
        --data_path bone_data/dataset_test.jsonl --image_root ./bone_data_mini \
        --output_file results/ablation/baseline_text_only_rawformat_FIXED.jsonl
"""

import argparse
import json
import os

import torch
from tqdm import tqdm

from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM

# ---------------------------------------------------------------------------
# The fix: patch generate() for this process only. Only the images-is-None
# branch is touched (the only branch proven buggy) -- the with-image branch is
# left exactly as-is since it hasn't been shown broken and is out of scope here.
# ---------------------------------------------------------------------------
_original_generate = LlavaMistralForCausalLM.generate


def _fixed_generate(self, inputs=None, images=None, image_sizes=None, **kwargs):
    if images is not None:
        return _original_generate(self, inputs=inputs, images=images,
                                   image_sizes=image_sizes, **kwargs)

    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if attention_mask is None:
        attention_mask = torch.ones_like(inputs)
    if "inputs_embeds" in kwargs:
        raise NotImplementedError("`inputs_embeds` is not supported")

    # FIX: pass input_ids straight through instead of pre-computing inputs_embeds.
    # forward() already embeds input_ids internally when inputs_embeds is None,
    # on every step of the generation loop -- this is the exact mechanism xattn's
    # own forward() relies on, and it is what removes the non-determinism.
    return super(LlavaMistralForCausalLM, self).generate(
        input_ids=inputs,
        position_ids=position_ids,
        attention_mask=attention_mask,
        **kwargs,
    )


LlavaMistralForCausalLM.generate = _fixed_generate

# Import AFTER patching so infer_prefix_style's internal `model.generate(...)`
# calls resolve to the patched method.
from run_inference import load_baseline_model
from run_inference_ablation import infer_prefix_style, parse_sample


def main():
    parser = argparse.ArgumentParser(description="Re-verify baseline text_only with the generate() fix")
    parser.add_argument("--prompt_style", default="conv", choices=["conv", "raw"])
    parser.add_argument("--model_path", default="microsoft/llava-med-v1.5-mistral-7b")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--conv_mode", default="mistral_instruct")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    print(f"Loading baseline model (generate() patched, prompt_style={args.prompt_style}) ...")
    tokenizer, model, image_processor = load_baseline_model(args.model_path)

    with open(args.data_path, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(samples)} test samples from {args.data_path}")

    n_empty = 0
    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for sample in tqdm(samples, desc=f"baseline-FIXED-{args.prompt_style}"):
            qid, image_file, question, reference, qa_type = parse_sample(sample)
            try:
                prediction = infer_prefix_style(
                    model, tokenizer, image_processor,
                    os.path.join(args.image_root, image_file),
                    question, qa_type, "text_only",
                    args.max_new_tokens, args.conv_mode, args.prompt_style,
                )
            except Exception as e:
                print(f"\n  [ERROR] qid={qid}: {e}")
                prediction = ""
            if not prediction.strip():
                n_empty += 1
            record = {
                "question_id": qid, "image": image_file, "question": question,
                "reference": reference, "prediction": prediction,
                "model_type": "baseline", "modality": "text_only",
                "prompt_style": args.prompt_style, "qa_type": qa_type,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone. {n_empty}/{len(samples)} empty predictions ({100*n_empty/len(samples):.1f}%).")
    print(f"Written to {args.output_file}")


if __name__ == "__main__":
    main()
