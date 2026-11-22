"""
Build RAG-augmented training dataset for Stage 3 fine-tuning.

For each training sample, retrieve the top-K most similar cases (same QA
type) from the rest of the training set, format them as few-shot context,
and prepend to the original question.  The augmented `question` field is
compatible with train_xattn.py without any modification.

Usage
-----
python build_rag_dataset.py \
    --stage2_model ./outputs_v2/stage2 \
    --base_model   microsoft/llava-med-v1.5-mistral-7b \
    --train_jsonl  data/dataset_train.jsonl \
    --image_root   ./bone_data \
    --output       data/dataset_train_rag.jsonl \
    --rag_index    ./rag_index \
    --rag_k        3

# Subsequent runs reuse the cached FAISS index (skip --rebuild_index).
"""

import argparse
import json
import os

import torch
from tqdm import tqdm

from rag_retriever import MultimodalRAGRetriever, detect_qa_type


# ---------------------------------------------------------------------------
# Load only the vision tower from the Stage 2 checkpoint
# ---------------------------------------------------------------------------

def load_vision_tower(model_path: str):
    from llava.model import LlavaMistralXAttnForCausalLM, LlavaMistralXAttnConfig

    print(f"Loading model from {model_path} ...")
    config = LlavaMistralXAttnConfig.from_pretrained(model_path)
    model = LlavaMistralXAttnForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    model.model.rebuild_hooks()

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device="cuda:0", dtype=torch.bfloat16)
    image_processor = vision_tower.image_processor

    model.eval()
    print("Vision tower ready.")
    return vision_tower, image_processor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build RAG-augmented training JSONL")
    parser.add_argument("--stage2_model",  required=True,
                        help="Path to Stage 2 checkpoint (e.g. ./outputs_v2/stage2)")
    parser.add_argument("--train_jsonl",   required=True,
                        help="Original training JSONL (e.g. data/dataset_train.jsonl)")
    parser.add_argument("--image_root",    required=True,
                        help="Root directory for images (e.g. ./bone_data)")
    parser.add_argument("--output",        required=True,
                        help="Output path (e.g. data/dataset_train_rag.jsonl)")
    parser.add_argument("--rag_index",     default="./rag_index",
                        help="Directory to save/load the FAISS index")
    parser.add_argument("--rag_k",         type=int,   default=3,
                        help="Number of retrieved cases to prepend")
    parser.add_argument("--rag_alpha",     type=float, default=0.6,
                        help="Visual weight in the combined ranking score")
    parser.add_argument("--rebuild_index", action="store_true",
                        help="Force rebuild the FAISS index even if cached")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ---- vision tower ----
    vision_tower, image_processor = load_vision_tower(args.stage2_model)
    device = "cuda:0"

    # ---- RAG index ----
    retriever = MultimodalRAGRetriever()
    index_exists = os.path.isfile(os.path.join(args.rag_index, "visual.faiss"))

    if index_exists and not args.rebuild_index:
        print(f"\nLoading cached RAG index from {args.rag_index} ...")
        retriever.load(args.rag_index)
    else:
        print(f"\nBuilding RAG index from {args.train_jsonl} ...")
        retriever.build(
            train_jsonl=args.train_jsonl,
            image_root=args.image_root,
            vision_tower=vision_tower,
            image_processor=image_processor,
            device=device,
            alpha=args.rag_alpha,
        )
        retriever.save(args.rag_index)

    # ---- load training samples ----
    with open(args.train_jsonl, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"\nLoaded {len(samples)} training samples from {args.train_jsonl}")

    # ---- augment each sample ----
    n_augmented  = 0
    n_no_context = 0
    n_skipped    = 0
    out_samples  = []

    for sample in tqdm(samples, desc="Augmenting"):
        question   = sample.get("question", "")
        image_file = sample.get("image", "")
        image_path = os.path.join(args.image_root, image_file)
        qa_type    = detect_qa_type(question)

        if not os.path.exists(image_path):
            out_samples.append(sample)
            n_skipped += 1
            continue

        try:
            # Fetch one extra so we can drop the self-match
            retrieved = retriever.retrieve(
                query_image_path=image_path,
                query_text=question,
                qa_type=qa_type,
                vision_tower=vision_tower,
                image_processor=image_processor,
                device=device,
                k=args.rag_k + 1,
            )
            # Drop the sample itself (same absolute image path)
            retrieved = [r for r in retrieved
                         if os.path.abspath(r["img_path"]) != os.path.abspath(image_path)]
            retrieved = retrieved[:args.rag_k]
            rag_context = retriever.format_context(retrieved, qa_type)
        except Exception as e:
            print(f"  [warn] {image_file}: {e}")
            rag_context = ""

        if rag_context:
            # Prepend few-shot context; keep same format as run_inference_rag.py
            augmented_question = f"{rag_context}\n\n{question}"
            n_augmented += 1
        else:
            augmented_question = question
            n_no_context += 1

        out_sample = dict(sample)
        out_sample["question"] = augmented_question
        out_samples.append(out_sample)

    # ---- save ----
    with open(args.output, "w", encoding="utf-8") as f:
        for s in out_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nDone.")
    print(f"  Total          : {len(out_samples)}")
    print(f"  Augmented      : {n_augmented}")
    print(f"  No RAG context : {n_no_context}")
    print(f"  Missing image  : {n_skipped}")
    print(f"  Saved to       : {args.output}")


if __name__ == "__main__":
    main()
