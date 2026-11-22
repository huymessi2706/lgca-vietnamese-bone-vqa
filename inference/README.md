# inference

Generation scripts for all three model types evaluated in the paper
(`xattn` = LGCA, `lora_ft` = LoRA baseline, `baseline` = frozen backbone, no
adaptation).

| Script | Use |
|---|---|
| `run_inference.py` | Single-model-type inference. Defines `load_baseline_model` / `load_lora_ft_model` / `load_xattn_model`, reused by `run_inference_ablation.py`. |
| `run_inference_ablation.py` | Adds `--modality {full,image_only,text_only}` for the modality-ablation tables. |
| `run_baseline_fixed_full.py`, `run_baseline_textonly_fixed.py`, `run_lora_ft_fixed.py` | **Use these, not the `baseline`/`lora_ft` paths in the two scripts above**, if you hit empty/garbled generations — see "Known inference bug" in the root [README](../README.md). These monkey-patch an `attention_mask` bug in `model_llava/llava/model/language_model/llava_mistral.py` without modifying the shared model code, and are the scripts actually used to produce this project's final baseline/LoRA-FT numbers. |

`infer_xattn()` is unaffected by that bug (it decodes by splitting on the
literal string `"Answer:"` rather than slicing by token position).
