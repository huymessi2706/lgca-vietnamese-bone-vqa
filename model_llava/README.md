# model_llava

The LGCA model architecture, adapted from the [LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
codebase.

```
model_llava/llava/
├── model/
│   ├── cross_attention.py                         LGCA cross-attention module (fixed α=0.05 gate)
│   ├── cross_attention_learned_gate.py             Learned-gate variant (cold-start control, see paper §V-I)
│   ├── anatomy_graph.py                            Exploratory anatomy-constrained-graph module (not used in the paper)
│   └── language_model/
│       ├── llava_mistral.py                        Base LLaVA-Med decoder (frozen-backbone baseline / LoRA host)
│       ├── llava_mistral_xattn.py                  LGCA decoder — reads LGCA_LAYOUT (see root README)
│       └── llava_mistral_xattn_learned_gate.py     LGCA decoder, learned-gate variant
├── eval/llm.py                                     Azure OpenAI judge harness (optional, unused for reported metrics)
├── serve/                                          Gradio demo server + example images (from upstream LLaVA)
├── conversation.py, mm_utils.py, constants.py, utils.py   Upstream LLaVA-Med plumbing (prompt templates, image preprocessing)
```

Everything here must be importable as `llava.*` — run training/inference scripts
with `model_llava/` on `PYTHONPATH`, or symlink `model_llava/llava/` next to the
script you are running. See the root [README](../README.md) for the
`LGCA_LAYOUT` environment variable, which controls the depth-placement
schedule read by `llava_mistral_xattn.py`.
