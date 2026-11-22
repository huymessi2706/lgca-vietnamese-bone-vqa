# data_prep

Dataset construction from raw hospital records. **No patient data ships
here** — these scripts document how the dataset was built and let you
reproduce the pipeline on your own equivalent records; see "Data (not
included)" in the root [README](../README.md) for the exact file layout
they expect.

| Script | Role |
|---|---|
| `preprocess_data.py` | Builds a patient-level train/val/test split from a raw hospital CSV export + image directory (`--train_ratio 0.75 --val_ratio 0.10 --test_ratio 0.15 --seed 42`). |
| `preprocess_merged.py` | More complete reconstruction of the official 8,798-sample dataset: merges CSV-parsing logic with multi-visit/multi-image detection. Start here if rebuilding the dataset from scratch. |
| `preprocess_enhanced.py`, `preprocess_balanced.py`, `preprocess_balanced_direct.py`, `preprocess_test.py` | Variant preprocessing passes used for specific ablations/diagnostics during the project (class-balancing, enhanced-context variants, smoke-test subsets). |
| `classify_patients.py` | Keyword-lexicon diagnostic-group classifier; also used by `metrics/proxy_classification_metrics.py` as the proxy-accuracy label source. |
| `build_minimal_test.py` | Builds a reduced/minimal test subset for fast iteration. |
| `augment_minority.py` | Minority-class augmentation helper. |

The resulting splits are referred to throughout the other folders as the
**reconstructed split** (6,807 train / 704 val / 1,287 test) — see the root
README for why this is distinct from the paper's original split and which
tables use which.
