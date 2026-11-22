# external_qwenvl

Code for the independently fine-tuned external competitor and the
English-language replication.

| Path | Role |
|---|---|
| `train_lora_qwenvl.py`, `run_inference_qwenvl.py` | Fine-tune/evaluate **Qwen2-VL-7B + LoRA** as the independently fine-tuned external competitor (paper §"External competitor"). A different backbone from LLaVA-Med, trained and evaluated under its own protocol. |
| `qwenvl_xattn/` | A from-scratch port of the LGCA cross-attention mechanism onto the Qwen2-VL backbone, used for the **English-language replication** — checks whether LGCA's relative ordering against LoRA holds on a different backbone/language. See `qwenvl_xattn/README.md` for details. Treat this as supplementary, capacity-confounded evidence only; it does not repeat the parameter-matched control reported for the main Vietnamese results. |
| `debug_qwenvl_labels.py` | Debugging helper used while building the Qwen2-VL training pipeline (label/tokenization alignment checks). |

See the root [README](../README.md) for how this comparison is framed
relative to the paper's primary (parameter-matched LoRA) evidence.
