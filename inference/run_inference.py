"""
Inference script for comparing LLaVA-Med baseline vs Stage 1 cross-attention model.

Both checkpoints are loaded from local disk (no HuggingFace download needed):
  - baseline : microsoft/llava-med-v1.5-mistral-7b  (HF cache on server)
  - xattn    : ./outputs/stage1                     (Stage 1 checkpoint on server)

Supported data formats
----------------------
LLaVA-Med eval format (data/eval/llava_med_eval_qa50_qa.jsonl):
  {"question_id": 0, "image": "foo.jpg", "text": "What is...?\\n<image>", "gpt4_answer": "..."}

Custom bone-data format (data/dataset_train.jsonl):
  {"image": "foo.jpg", "question": "...", "answer": "..."}

Usage
-----
# Baseline (original LLaVA-Med, no cross-attention training)
python run_inference.py \\
    --model_type baseline \\
    --model_path microsoft/llava-med-v1.5-mistral-7b \\
    --data_path  data/eval/llava_med_eval_qa50_qa.jsonl \\
    --image_root ./images \\
    --output_file results/baseline_outputs.jsonl

# XAttn Stage 1 (cross-attention alignment checkpoint)
python run_inference.py \\
    --model_type  xattn \\
    --model_path  ./outputs/stage1 \\
    --base_model  microsoft/llava-med-v1.5-mistral-7b \\
    --data_path   data/eval/llava_med_eval_qa50_qa.jsonl \\
    --image_root  ./images \\
    --output_file results/xattn_stage1_outputs.jsonl

# LoRA fine-tuned baseline (ablation: fine-tune without cross-attn)
python run_inference.py \\
    --model_type  lora_ft \\
    --model_path  ./outputs_v2/lora_baseline \\
    --base_model  microsoft/llava-med-v1.5-mistral-7b \\
    --data_path   data/dataset_test.jsonl \\
    --image_root  ./bone_data \\
    --output_file results/lora_ft_outputs.jsonl
"""

import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_baseline_model(model_path: str):
    """Load original LLaVA-Med via standard builder (uses HF local cache)."""
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name
    )
    model.eval()
    print(f"[Baseline] Loaded: {model_name}")
    return tokenizer, model, image_processor


def load_lora_ft_model(model_path: str, base_model: str):
    """Load LoRA fine-tuned model.

    Two cases:
      A) Checkpoint dir has adapter_config.json → load base + PEFT adapter + merge.
      B) No adapter_config.json (train_lora_baseline saved merged model directly)
         → load the merged LlavaMistralForCausalLM straight from model_path.
    """
    import glob
    from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM

    # Find the adapter/checkpoint directory to inspect
    checkpoint_dirs = sorted(glob.glob(os.path.join(model_path, "checkpoint-*")))
    adapter_path = checkpoint_dirs[-1] if checkpoint_dirs else model_path

    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        # Case A: PEFT adapter saved by Trainer — apply onto fresh base model
        from peft import PeftModel
        print(f"[LoRA-FT] Loading base model: {base_model}")
        model = LlavaMistralForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
        )
        print(f"[LoRA-FT] Applying PEFT adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        tokenizer_path = base_model
    else:
        # Case B: merged model already saved by train_lora_baseline.merge_and_unload()
        print(f"[LoRA-FT] Loading merged model from: {model_path}")
        model = LlavaMistralForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
        )
        tokenizer_path = model_path  # tokenizer was saved alongside merged weights

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device="cuda:0", dtype=torch.bfloat16)
    image_processor = vision_tower.image_processor

    model.eval()
    print(f"[LoRA-FT] Ready.")
    return tokenizer, model, image_processor


def load_xattn_model(checkpoint_path: str, base_model: str, load_in_8bit: bool = False):
    """Load Stage 1 cross-attention model from local checkpoint directory.

    load_in_8bit must match how the checkpoint was trained.  If the model was
    trained with --load_in_8bit the saved layers are Linear8bitLt (bitsandbytes).
    Loading them without load_in_8bit=True causes 'Parameter has no attribute CB'.
    """
    from llava.model import LlavaMistralXAttnForCausalLM, LlavaMistralXAttnConfig

    config = LlavaMistralXAttnConfig.from_pretrained(checkpoint_path)

    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        model = LlavaMistralXAttnForCausalLM.from_pretrained(
            checkpoint_path,
            config=config,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
        )
    else:
        model = LlavaMistralXAttnForCausalLM.from_pretrained(
            checkpoint_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    # Tokenizer lives in the base model (checkpoint only stores weights)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and prepare vision tower
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    # load_model() places weights on CPU by default — must move to GPU explicitly
    # before forward pass, otherwise pixel_values (cuda) vs vision_tower (cpu) mismatch
    llm_device = next(model.model.embed_tokens.parameters()).device
    vision_tower.to(device=llm_device, dtype=torch.bfloat16)
    image_processor = vision_tower.image_processor

    # Re-attach forward hooks (not serialized in checkpoint)
    model.model.rebuild_hooks()

    # Cross-attn modules were kept in Float32 during training to avoid AMP unscale_ crash.
    # Cast them to bfloat16 now so they match the LLM hidden-state dtype at inference time.
    for module in model.model.cross_attn_modules:
        module.to(dtype=torch.bfloat16)

    # Suppress the 3-step diagnostic prints during inference
    model._diag_step = 999
    model.eval()
    print(f"[XAttn] Loaded Stage 1 checkpoint: {checkpoint_path}")
    return tokenizer, model, image_processor


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_sample(line: dict) -> tuple:
    """
    Returns (question_id, image_filename, question_text, reference_answer).
    Handles both LLaVA-Med eval format and custom bone-data format.
    """
    if "text" in line:
        # LLaVA-Med eval format: question is in "text", image token is embedded
        question = line["text"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
        reference = line.get("gpt4_answer", "")
    else:
        # Custom format: separate question and answer fields
        question = line.get("question", "")
        reference = line.get("answer", "")

    qid = line.get("question_id", line.get("id", 0))
    return qid, line["image"], question, reference


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------

@torch.inference_mode()
def infer_lora_ft(
    model, tokenizer, image_processor,
    image_path: str, question: str, max_new_tokens: int,
    conv_mode: str = "mistral_instruct",
    min_new_tokens: int = 0,
) -> str:
    """Run inference with the merged LoRA fine-tuned model.

    Uses mistral_instruct conversation format — matches LoRAVQADataset training
    format and the base Mistral Instruct model's generation conditioning.

    min_new_tokens > 0 forbids EOS for the first N tokens — diagnostic only
    (see what the EOS-degenerate adapter "would" say if forced to continue).
    Do NOT use this for the official metrics table: it changes the decoding
    protocol relative to every other configuration (baseline/XAttn/LoRA v1),
    which were all scored with greedy decoding and no forced minimum length.
    """
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    image = Image.open(image_path).convert("RGB")
    image_tensor = (
        image_processor(images=image, return_tensors="pt")
        .pixel_values
        .to(dtype=torch.bfloat16)
        .cuda()
    )

    stop_str = conv.sep2 if conv.sep2 else conv.sep
    stopping_criteria = KeywordsStoppingCriteria(
        [stop_str] if stop_str else [], tokenizer, input_ids
    )

    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        stopping_criteria=[stopping_criteria] if stop_str else [],
    )

    # images is always passed here, so LlavaMistralForCausalLM.generate()
    # builds inputs_embeds internally and HF generate() returns ONLY the
    # newly generated tokens (not input+generated as with plain input_ids
    # decoding) -- do NOT slice off input_ids.shape[1] here.
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if stop_str and answer.endswith(stop_str):
        answer = answer[: -len(stop_str)].strip()
    return answer


@torch.inference_mode()
def infer_baseline(
    model, tokenizer, image_processor,
    image_path: str, question: str, max_new_tokens: int,
    conv_mode: str = "mistral_instruct",
) -> str:
    """Run inference with the original LLaVA-Med (projection prefix approach)."""
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.mm_utils import tokenizer_image_token, process_images, KeywordsStoppingCriteria

    # Build prompt using LLaVA conversation template
    qs = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)[0]
    image_tensor = image_tensor.unsqueeze(0).half().cuda()

    # For mistral_instruct (SeparatorStyle.LLAMA_2), stop on </s>
    stop_str = conv.sep2 if conv.sep2 else conv.sep
    stopping_criteria = KeywordsStoppingCriteria(
        [stop_str] if stop_str else [], tokenizer, input_ids
    )

    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        stopping_criteria=[stopping_criteria] if stop_str else [],
    )
    # images is always passed here too, so LlavaMistralForCausalLM.generate()
    # builds inputs_embeds internally (see prepare_inputs_labels_for_multimodal
    # in llava_mistral.py) and HF generate() returns ONLY the newly generated
    # tokens -- do NOT slice off input_ids.shape[1] here (this was the source
    # of the catastrophic near-empty-prediction bug found and fixed in the
    # lora_ft/xattn inference paths; infer_baseline uses the same generate()
    # override and has the same issue).
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if stop_str and answer.endswith(stop_str):
        answer = answer[: -len(stop_str)].strip()
    return answer


@torch.inference_mode()
def infer_xattn(
    model, tokenizer, image_processor,
    image_path: str, question: str, max_new_tokens: int,
) -> str:
    """
    Run inference with the XAttn Stage 1 model.

    Unlike the baseline, visual context reaches the LLM through cross-attention hooks
    (not as a text prefix). The prompt keeps the <image> token as a text marker
    (consistent with training) but it's never inserted as IMAGE_TOKEN_INDEX (-200).
    """
    prompt = f"{DEFAULT_IMAGE_TOKEN}\n{question}\nAnswer:"
    input_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).input_ids.cuda()

    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(dtype=torch.bfloat16).cuda()

    output_ids = model.generate(
        inputs=input_ids,
        images=pixel_values,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )

    # model.generate() uses inputs_embeds internally; the returned token IDs
    # correspond to the generated tokens only (not the prompt tokens).
    # However we guard with the "Answer:" split in case full sequence is returned.
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if "Answer:" in decoded:
        decoded = decoded.split("Answer:")[-1].strip()
    return decoded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run inference for model comparison")
    parser.add_argument("--model_type",  required=True, choices=["baseline", "xattn", "lora_ft"],
                        help="'baseline': original LLaVA-Med | 'xattn': XAttn checkpoint | "
                             "'lora_ft': LoRA fine-tuned ablation baseline")
    parser.add_argument("--model_path",  required=True,
                        help="HF model ID (baseline) or local checkpoint path (xattn)")
    parser.add_argument("--base_model",  default="microsoft/llava-med-v1.5-mistral-7b",
                        help="Base model for tokenizer (xattn only; must be the original HF model ID)")
    parser.add_argument("--data_path",   required=True,
                        help="Test JSONL file (LLaVA-Med eval format or custom bone-data format)")
    parser.add_argument("--image_root",  required=True,
                        help="Root directory containing the test images")
    parser.add_argument("--output_file", required=True,
                        help="Output JSONL file (one prediction per line)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate per answer")
    parser.add_argument("--conv_mode",   default="mistral_instruct",
                        help="Conversation template for baseline model (default: mistral_instruct)")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load xattn checkpoint in 8-bit. Required when the checkpoint was "
                             "trained with --load_in_8bit (otherwise 'Parameter has no CB' error).")
    parser.add_argument("--min_new_tokens", type=int, default=0,
                        help="lora_ft only. Diagnostic flag: forbid EOS for the first N tokens, "
                             "to inspect what an EOS-degenerate adapter would say if forced to "
                             "continue. Default 0 = official protocol (greedy, no forced length); "
                             "do NOT set >0 when producing numbers for the metrics comparison table.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    # ---- load model ----
    print(f"\nLoading {args.model_type} model ...")
    if args.model_type == "baseline":
        tokenizer, model, image_processor = load_baseline_model(args.model_path)
        def infer_fn(img, q):
            return infer_baseline(model, tokenizer, image_processor, img, q,
                                  args.max_new_tokens, args.conv_mode)
    elif args.model_type == "lora_ft":
        tokenizer, model, image_processor = load_lora_ft_model(args.model_path, args.base_model)
        def infer_fn(img, q):
            return infer_lora_ft(model, tokenizer, image_processor, img, q, args.max_new_tokens,
                                  min_new_tokens=args.min_new_tokens)
    else:
        tokenizer, model, image_processor = load_xattn_model(
            args.model_path, args.base_model, load_in_8bit=args.load_in_8bit
        )
        def infer_fn(img, q):
            return infer_xattn(model, tokenizer, image_processor, img, q, args.max_new_tokens)

    # ---- load test data ----
    with open(args.data_path) as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(samples)} test samples from {args.data_path}")

    # ---- inference loop ----
    skipped, errors = 0, 0
    with open(args.output_file, "w") as out_f:
        for sample in tqdm(samples, desc=f"Inferring ({args.model_type})"):
            qid, image_file, question, reference = parse_sample(sample)
            image_path = os.path.join(args.image_root, image_file)

            if not os.path.exists(image_path):
                skipped += 1
                continue

            try:
                prediction = infer_fn(image_path, question)
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
                "domain":      sample.get("domain", {}),
                "type":        sample.get("type", ""),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    total = len(samples)
    print(f"\nDone.  Processed {total - skipped - errors}/{total} samples.")
    if skipped:
        print(f"  Skipped {skipped} samples (image not found)")
    if errors:
        print(f"  Errors  {errors} samples (see messages above)")
    print(f"Results saved → {args.output_file}")


if __name__ == "__main__":
    main()
