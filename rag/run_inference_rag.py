"""
RAG-augmented Inference Script — XAttn model only
==================================================
Identical flow to run_inference.py for the xattn model, but each
generation prompt is prepended with top-K retrieved training cases
(same QA type, ranked by combined visual + text similarity).

The baseline path (run_inference.py) is NOT touched.

Usage
-----
# Step 1 — build and cache the RAG index (once, ~5-10 min)
python run_inference_rag.py \\
    --model_type  xattn \\
    --model_path  ./outputs/stage1 \\
    --base_model  microsoft/llava-med-v1.5-mistral-7b \\
    --data_path   data/dataset_test.jsonl \\
    --image_root  ./bone_data \\
    --output_file results/xattn_rag_outputs.jsonl \\
    --rag_train   data/dataset_train.jsonl \\
    --rag_index   ./rag_index \\
    --rag_k       3 \\
    --rag_alpha   0.6

# Step 2 — subsequent runs reuse the saved index (fast)
# Same command: if ./rag_index already exists, it loads instantly.
"""

import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from rag_retriever import MultimodalRAGRetriever, detect_qa_type


# ---------------------------------------------------------------------------
# Model loading  (same as run_inference.py)
# ---------------------------------------------------------------------------

def load_xattn_model(checkpoint_path: str, base_model: str):
    from llava.model import LlavaMistralXAttnForCausalLM, LlavaMistralXAttnConfig

    config = LlavaMistralXAttnConfig.from_pretrained(checkpoint_path)
    model = LlavaMistralXAttnForCausalLM.from_pretrained(
        checkpoint_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    llm_device = next(model.model.embed_tokens.parameters()).device
    vision_tower.to(device=llm_device, dtype=torch.bfloat16)
    image_processor = vision_tower.image_processor

    model.model.rebuild_hooks()

    # Cross-attn modules were kept in Float32 during training; cast to bfloat16 for inference.
    for module in model.model.cross_attn_modules:
        module.to(dtype=torch.bfloat16)

    model._diag_step = 999
    model.eval()
    print(f"[XAttn] Loaded: {checkpoint_path}")
    return tokenizer, model, image_processor


# ---------------------------------------------------------------------------
# RAG-augmented inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def infer_xattn_rag(
    model,
    tokenizer,
    image_processor,
    image_path: str,
    question: str,
    max_new_tokens: int,
    retriever: MultimodalRAGRetriever,
    vision_tower,
    device: str = "cuda",
    rag_k: int = 3,
) -> str:
    qa_type = detect_qa_type(question)

    # Retrieve similar cases
    try:
        retrieved = retriever.retrieve(
            query_image_path=image_path,
            query_text=question,
            qa_type=qa_type,
            vision_tower=vision_tower,
            image_processor=image_processor,
            device=device,
            k=rag_k,
        )
        rag_context = retriever.format_context(retrieved, qa_type)
    except Exception as e:
        # Retrieval failure is non-fatal — fall back to no-context generation
        rag_context = ""
        print(f"    [RAG warn] retrieval failed: {e}")

    # Build augmented prompt
    # Token budget: max_length=512 — keep RAG context ≤ 150 tokens (~600 chars)
    if rag_context:
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{rag_context}\n\n{question}\nAnswer:"
    else:
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{question}\nAnswer:"

    input_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).input_ids.to(device)

    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(dtype=torch.bfloat16, device=device)

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
# Data loading  (same helper as run_inference.py)
# ---------------------------------------------------------------------------

def parse_sample(line: dict) -> tuple:
    if "text" in line:
        question  = line["text"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
        reference = line.get("gpt4_answer", "")
    else:
        question  = line.get("question", "")
        reference = line.get("answer", "")
    qid = line.get("question_id", line.get("id", 0))
    return qid, line["image"], question, reference


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG-augmented XAttn inference")
    parser.add_argument("--model_type",      required=True, choices=["xattn"],
                        help="Only xattn is supported in this script")
    parser.add_argument("--model_path",      required=True)
    parser.add_argument("--base_model",      default="microsoft/llava-med-v1.5-mistral-7b")
    parser.add_argument("--data_path",       required=True)
    parser.add_argument("--image_root",      required=True)
    parser.add_argument("--output_file",     required=True)
    parser.add_argument("--max_new_tokens",  type=int, default=256)
    # RAG-specific
    parser.add_argument("--rag_train",       required=True,
                        help="Path to training JSONL used to build the retrieval index")
    parser.add_argument("--rag_index",       default="./rag_index",
                        help="Directory to save/load the pre-built FAISS index")
    parser.add_argument("--rag_k",           type=int,   default=3,
                        help="Number of retrieved cases to inject as few-shot context")
    parser.add_argument("--rag_alpha",       type=float, default=0.6,
                        help="Weight for visual similarity (1-alpha = text weight)")
    parser.add_argument("--rebuild_index",   action="store_true",
                        help="Force rebuild the index even if it already exists on disk")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    # ---- Load model ----
    print("\nLoading xattn model ...")
    tokenizer, model, image_processor = load_xattn_model(args.model_path, args.base_model)
    vision_tower = model.get_vision_tower()
    device = str(next(model.model.embed_tokens.parameters()).device)

    # ---- Build or load RAG index ----
    retriever = MultimodalRAGRetriever()
    index_exists = os.path.isfile(os.path.join(args.rag_index, "visual.faiss"))

    if index_exists and not args.rebuild_index:
        print(f"\nLoading cached RAG index from {args.rag_index} ...")
        retriever.load(args.rag_index)
    else:
        print(f"\nBuilding RAG index from {args.rag_train} ...")
        retriever.build(
            train_jsonl=args.rag_train,
            image_root=args.image_root,
            vision_tower=vision_tower,
            image_processor=image_processor,
            device=device,
            alpha=args.rag_alpha,
        )
        retriever.save(args.rag_index)

    # ---- Load test data ----
    with open(args.data_path, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(samples)} test samples from {args.data_path}")

    # ---- Inference loop ----
    skipped, errors = 0, 0
    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for sample in tqdm(samples, desc="Inferring (xattn+RAG)"):
            qid, image_file, question, reference = parse_sample(sample)
            image_path = os.path.join(args.image_root, image_file)

            if not os.path.exists(image_path):
                skipped += 1
                continue

            try:
                prediction = infer_xattn_rag(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    image_path=image_path,
                    question=question,
                    max_new_tokens=args.max_new_tokens,
                    retriever=retriever,
                    vision_tower=vision_tower,
                    device=device,
                    rag_k=args.rag_k,
                )
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
                "model_type":  "xattn_rag",
                "rag_k":       args.rag_k,
                "domain":      sample.get("domain", {}),
                "type":        sample.get("type", ""),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    total = len(samples)
    print(f"\nDone.  Processed {total - skipped - errors}/{total} samples.")
    if skipped: print(f"  Skipped {skipped} (image not found)")
    if errors:  print(f"  Errors  {errors} (see above)")
    print(f"Results → {args.output_file}")


if __name__ == "__main__":
    main()
