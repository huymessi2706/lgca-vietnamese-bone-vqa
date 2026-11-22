"""
Evaluate and compare two model outputs using standard NLP metrics.

Metrics
-------
  BLEU-1/2/4  : n-gram precision (measures exact term overlap)
  ROUGE-1/2/L : recall-oriented overlap (measures coverage of reference)
  METEOR      : synonym/stem-aware alignment (better for paraphrases)
  BERTScore   : contextual embedding similarity via BiomedBERT
                (most meaningful for medical text where synonyms matter)

Install dependencies (once, on server):
  pip install rouge-score bert-score nltk

Usage
-----
# Compare baseline vs XAttn Stage 1
python compute_metrics.py \\
    --baseline_file results/baseline_outputs.jsonl \\
    --xattn_file    results/xattn_stage1_outputs.jsonl \\
    --output_file   results/metrics_comparison.json \\
    --bert_model    dmis-lab/biobert-base-cased-v1.2

# If BiomedBERT is not cached, fall back to the standard model:
#   --bert_model roberta-large
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer as rs
import bert_score as bscore


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def ensure_nltk_data():
    for resource, path in [
        ("punkt_tab",    "tokenizers/punkt_tab"),
        ("wordnet",      "corpora/wordnet"),
        ("omw-1.4",      "corpora/omw-1.4"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    """Load JSONL predictions by line order (question_id may not be unique)."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def align_samples(files: Dict[str, List[dict]]) -> Tuple[List[str], Dict[str, List[str]], List[dict]]:
    """
    Align multiple prediction files by line index.
    Returns (references, {model_name: predictions}, metadata).
    """
    n = min(len(v) for v in files.values())
    first_key = next(iter(files))
    references = [files[first_key][i]["reference"] for i in range(n)]
    metadata   = [files[first_key][i]              for i in range(n)]
    predictions = {name: [records[i]["prediction"] for i in range(n)]
                   for name, records in files.items()}
    return references, predictions, metadata


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    return nltk.word_tokenize(text.lower())


def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    smooth    = SmoothingFunction().method1
    refs_tok  = [[tokenize(r)] for r in references]
    preds_tok = [tokenize(p)   for p in predictions]
    return {
        "bleu1": corpus_bleu(refs_tok, preds_tok, weights=(1,0,0,0),               smoothing_function=smooth),
        "bleu2": corpus_bleu(refs_tok, preds_tok, weights=(0.5,0.5,0,0),           smoothing_function=smooth),
        "bleu4": corpus_bleu(refs_tok, preds_tok, weights=(0.25,0.25,0.25,0.25),   smoothing_function=smooth),
    }


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals: Dict[str, float] = defaultdict(float)
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for k, v in scores.items():
            totals[k] += v.fmeasure
    n = len(predictions)
    return {k: v / n for k, v in totals.items()}


def compute_meteor(predictions: List[str], references: List[str]) -> Dict[str, float]:
    scores = [
        meteor_score([tokenize(ref)], tokenize(pred))
        for pred, ref in zip(predictions, references)
    ]
    return {"meteor": sum(scores) / len(scores)}


def compute_bertscore(
    predictions: List[str],
    references:  List[str],
    model_type:  str,
    device:      str = "cuda",
) -> Dict[str, float]:
    from bert_score.utils import model2layers
    num_layers = model2layers.get(model_type, 12)
    P, R, F = bscore.score(
        predictions, references,
        model_type=model_type,
        num_layers=num_layers,
        lang="en",
        verbose=False,
        device=device,
    )
    return {
        "bertscore_P":  P.mean().item(),
        "bertscore_R":  R.mean().item(),
        "bertscore_F1": F.mean().item(),
    }


def compute_all(
    predictions: List[str],
    references:  List[str],
    bert_model:  str,
) -> Dict[str, float]:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    metrics: Dict[str, float] = {}
    print("  Computing BLEU  ...", end=" ", flush=True)
    metrics.update(compute_bleu(predictions, references));    print("done")
    print("  Computing ROUGE ...", end=" ", flush=True)
    metrics.update(compute_rouge(predictions, references));   print("done")
    print("  Computing METEOR...", end=" ", flush=True)
    metrics.update(compute_meteor(predictions, references));  print("done")
    print(f"  Computing BERTScore ({bert_model}) ...", end=" ", flush=True)
    metrics.update(compute_bertscore(predictions, references, bert_model, device))
    print("done")
    return metrics


# ---------------------------------------------------------------------------
# Per-domain breakdown
# ---------------------------------------------------------------------------

DOMAINS = ["chest_xray", "mri", "ct_scan", "histology", "gross"]


def per_domain_metrics(
    baseline_preds: List[str],
    xattn_preds:    List[str],
    references:     List[str],
    metadata:       List[dict],
    bert_model:     str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute ROUGE-L and BERTScore-F1 per imaging domain."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    buckets: Dict[str, Dict[str, list]] = {
        d: {"refs": [], "base": [], "xattn": []} for d in DOMAINS
    }
    buckets["other"] = {"refs": [], "base": [], "xattn": []}

    for base_p, xattn_p, ref, meta in zip(baseline_preds, xattn_preds, references, metadata):
        domain_flags = meta.get("domain", {})
        assigned = False
        for d in DOMAINS:
            if domain_flags.get(d, False):
                buckets[d]["refs"].append(ref)
                buckets[d]["base"].append(base_p)
                buckets[d]["xattn"].append(xattn_p)
                assigned = True
                break
        if not assigned:
            buckets["other"]["refs"].append(ref)
            buckets["other"]["base"].append(base_p)
            buckets["other"]["xattn"].append(xattn_p)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    scorer  = rs.RougeScorer(["rougeL"], use_stemmer=True)

    for domain, data in buckets.items():
        if not data["refs"]:
            continue
        n = len(data["refs"])

        def rouge_l(preds, refs):
            return sum(scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)) / n

        def bscore_f1(preds, refs):
            _, _, F = bscore.score(preds, refs, model_type=bert_model, lang="en",
                                   verbose=False, device=device)
            return F.mean().item()

        results[domain] = {
            "n_samples": n,
            "baseline":  {"rougeL": rouge_l(data["base"],  data["refs"]),
                          "bertscore_F1": bscore_f1(data["base"],  data["refs"])},
            "xattn_s1":  {"rougeL": rouge_l(data["xattn"], data["refs"]),
                          "bertscore_F1": bscore_f1(data["xattn"], data["refs"])},
        }
    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "bleu1":        "BLEU-1",
    "bleu2":        "BLEU-2",
    "bleu4":        "BLEU-4",
    "rouge1":       "ROUGE-1",
    "rouge2":       "ROUGE-2",
    "rougeL":       "ROUGE-L",
    "meteor":       "METEOR",
    "bertscore_P":  "BERTScore-P",
    "bertscore_R":  "BERTScore-R",
    "bertscore_F1": "BERTScore-F1",
}

W = 20   # column width


def print_comparison_table(all_metrics: Dict[str, Dict[str, float]]):
    model_names = list(all_metrics.keys())
    col_w = 12
    header = f"{'Metric':<{W}}" + "".join(f"{n:>{col_w}}" for n in model_names)
    print("\n" + "=" * (W + col_w * len(model_names)))
    print(header)
    print("-" * (W + col_w * len(model_names)))
    baseline_key = model_names[0]
    for key, label in METRIC_LABELS.items():
        if key not in all_metrics[baseline_key]:
            continue
        row = f"{label:<{W}}"
        base_val = all_metrics[baseline_key][key]
        for i, name in enumerate(model_names):
            val = all_metrics[name][key]
            if i == 0:
                row += f"{val:>{col_w}.4f}"
            else:
                delta = val - base_val
                sign = "+" if delta >= 0 else ""
                row += f"{val:>{col_w}.4f}"
        print(row)
    print("=" * (W + col_w * len(model_names)) + "\n")


def print_domain_table(domain_results: Dict):
    if not domain_results:
        return
    print("\n--- Per-domain breakdown (ROUGE-L / BERTScore-F1) ---")
    print(f"{'Domain':<14} {'N':>4}  {'Base RL':>8} {'S1 RL':>8} {'ΔRL':>8}  {'Base BS':>8} {'S1 BS':>8} {'ΔBS':>8}")
    print("-" * 72)
    for domain, data in sorted(domain_results.items()):
        n   = data["n_samples"]
        brl = data["baseline"]["rougeL"]
        xrl = data["xattn_s1"]["rougeL"]
        bbs = data["baseline"]["bertscore_F1"]
        xbs = data["xattn_s1"]["bertscore_F1"]
        drl_s = ("+" if xrl - brl >= 0 else "") + f"{xrl - brl:.4f}"
        dbs_s = ("+" if xbs - bbs >= 0 else "") + f"{xbs - bbs:.4f}"
        print(f"{domain:<14} {n:>4}  {brl:>8.4f} {xrl:>8.4f} {drl_s:>8}  {bbs:>8.4f} {xbs:>8.4f} {dbs_s:>8}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics for model comparison")
    parser.add_argument("--files", nargs="+", required=True,
                        help="JSONL result files in order: baseline stage1 stage2 ...")
    parser.add_argument("--names", nargs="+",
                        help="Display names for each file (same order as --files)")
    parser.add_argument("--output_file", default="results/metrics_comparison.json")
    parser.add_argument("--bert_model",  default="dmis-lab/biobert-base-cased-v1.2")
    args = parser.parse_args()

    names = args.names if args.names else [os.path.splitext(os.path.basename(f))[0] for f in args.files]
    assert len(names) == len(args.files), "--names must have same count as --files"

    ensure_nltk_data()

    # ---- load & align ----
    files_data = {}
    for name, path in zip(names, args.files):
        print(f"Loading {name}: {path}")
        files_data[name] = load_jsonl(path)

    refs, predictions, metadata = align_samples(files_data)
    print(f"Aligned {len(refs)} samples.\n")

    # ---- compute metrics for each model ----
    all_metrics = {}
    for name in names:
        print(f"[ {name} ]")
        all_metrics[name] = compute_all(predictions[name], refs, args.bert_model)
        print()

    # ---- display ----
    print_comparison_table(all_metrics)

    # ---- save ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    baseline_name = names[0]
    output = {
        "num_samples": len(refs),
        "bert_model":  args.bert_model,
        "models":      all_metrics,
        "delta_vs_baseline": {
            name: {k: all_metrics[name][k] - all_metrics[baseline_name][k]
                   for k in all_metrics[baseline_name]}
            for name in names[1:]
        },
    }
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Metrics saved → {args.output_file}")


if __name__ == "__main__":
    main()
