# metrics

All metric computation used in the paper.

| Script | Computes |
|---|---|
| `proxy_classification_metrics.py` | Keyword-lexicon-based diagnostic-group proxy accuracy + laterality side-matching (reuses `data_prep/classify_patients.py`). The paper's primary accuracy-style metric; explicitly **not** a clinically validated correctness measure — see the paper's Discussion. |
| `compute_metrics.py` | BLEU-1/2/4, ROUGE-1/2/L, METEOR, BERTScore-P/R/F1. See the root [README](../README.md) for a required monkey-patch when using `dmis-lab/biobert-base-cased-v1.2` (the model used for the paper's reported BERT-F1 numbers) under recent `transformers` versions. |
| `compute_metrics_clean_subset.py` | Same as above, but splits input into leak/clean/overall subsets first. Will raise `ZeroDivisionError` on an already-fully-clean file — use `compute_metrics.py` directly in that case. |
| `compute_metric_en.py` | English-replication counterpart of `compute_metrics.py`. |
| `leakage_audit.py` | Flags a sample as leaked if its reference answer appears verbatim, or ≥85%-token-contained, in its own question text — reproduces the leakage counts reported in the paper's leakage-audit section. |
| `bootstrap_ci.py` | Patient-clustered bootstrap significance test, used for the external-competitor (Qwen2-VL+LoRA) comparison. |
| `plot_metrics.py` | Plotting helper for metric comparison figures. |

See the root README's "Reproducing each table" section for exact invocation
of each script.
