# rag

Retrieval-augmented generation (RAG) extension evaluated as a supplementary
experiment in the paper.

| Script | Role |
|---|---|
| `build_rag_index.py`, `build_clip_index.py` | Build BM25 / CLIP retrieval indices over the training set. |
| `rag_retriever.py` | Queries the built indices at inference time. |
| `build_rag_dataset.py`, `build_rag_train_data.py` | Build RAG-aware training data (retrieved cases concatenated into the training prompt). |
| `build_rag_train_data_strict.py` | Stricter variant that additionally excludes same-patient and near-duplicate-answer candidates from the retrieval context — used for the re-run discussed in the paper's Discussion section on RAG-aware training-set contamination. |
| `run_inference_rag.py`, `run_rag_inference.py` | Pre-RAG / Post-RAG inference. |

**Finding reported in the paper**: none of the BM25/CLIP configurations
tested improved BERTScore over the no-retrieval baseline on either split
tested. Report this as "the specific configurations tested did not help,"
not as a general finding against retrieval augmentation — see the root
[README](../README.md).
