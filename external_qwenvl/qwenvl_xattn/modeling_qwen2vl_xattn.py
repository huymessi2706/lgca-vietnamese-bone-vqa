"""
Qwen2-VL + Layered Gated Cross-Attention (thesis extension #2)
================================================================

Motivation
----------
The thesis's main result (LLaVA-Med) shows cross-attention beats token
concatenation by +0.865 BERTScore-F1 — but LLaVA-Med's base LLM (Mistral-7B)
has a severe Vietnamese-language gap, so that comparison conflates two
effects: "fixing a broken base model" and "fusion architecture quality".

Qwen2-VL is the cleanest possible test to isolate the second effect alone:
its base LLM is already multilingual-capable (zero-shot BERTScore 0.86 on
Vietnamese, no fine-tuning at all — see results/all_metrics.json). If
cross-attention *also* helps on top of a base model that doesn't have a
language gap to fix, that is strong evidence the fusion mechanism itself is
generically better than concatenation, not just a rescue mechanism for weak
base models.

Design — what changes vs. the original Qwen2-VL concat architecture
---------------------------------------------------------------------
Original Qwen2-VL: image patches (after the ViT + PatchMerger, already
projected into the *same* hidden size as the text decoder) are spliced
directly into the input embedding sequence at <|image_pad|> token positions.
Self-attention then has to process a much longer sequence (text + ~256-1024
image tokens) and visual info is anchored only at those fixed positions.

This file: image tokens are *not* placed in the input sequence at all. They
are kept out of self-attention entirely and re-injected at multiple decoder
layers via GatedCrossAttention (text queries, image keys/values), exactly
mirroring llava/model/language_model/llava_mistral_xattn.py's design:
  - bottom third of decoder layers : self-attention only (no cross-attn)
  - middle third                   : sparse cross-attn (top-k) every 2 layers
  - top third                      : dense cross-attn every layer
  - alpha=0.05 fixed gate + zero-init out_proj (same stability trick as
    LLaVA's XAttn — see cross_attention.py for the gradient-vanishing
    rationale for NOT using a learned gate)

Because Qwen2-VL's vision tower already projects patches into the text
hidden size (PatchMerger's output dim == decoder hidden_size), vision_dim ==
text_dim here — simpler than LLaVA's case (CLIP 1024 -> Mistral 4096). This
also means cross-attn module structure can be built directly in __init__
from config alone (no two-step "build after vision tower loads" dance like
LLaVA needed).

We also skip the learnable 2D VisualPositionEmbedding that LLaVA's
GatedCrossAttention supports (pass num_patches=None): Qwen2-VL's own ViT
already applies rotary position embeddings to patches before the merger, so
the tokens we receive are already spatially aware. Re-adding a *fixed-grid*
positional embedding would conflict with Qwen2-VL's variable image
resolution (dynamic patch count per image, not a fixed 576 like CLIP).

KNOWN RISK / UNTESTED — read before running
--------------------------------------------
This was written by inspecting transformers 5.7.0's modeling_qwen2_vl.py
(locally available) and train_lora_qwenvl.py's working LoRA recipe (which
ran against whatever transformers version is in the server's "qwenvl" conda
env). Two transformers internals are known to have changed across versions
and are handled defensively below, but NOT verified against your exact
server version yet:
  1. `model.model.language_model.layers` (newer) vs `model.model.layers`
     (older) — see `_get_text_decoder()`.
  2. `self.visual(pixel_values, grid_thw=...)` returning a tensor directly
     (older) vs a ModelOutput with `.pooler_output` (newer) — see
     `extract_visual_tokens()`.
Expect to need 1-2 rounds of fixing AttributeError/TypeError against your
actual env, the same way train_lora_baseline_v2/v3 needed environment fixes.
Run a quick smoke test (a single forward pass on 1 sample) before launching
a full training job.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from transformers import Qwen2VLForConditionalGeneration

from cross_attention import GatedCrossAttention


# ---------------------------------------------------------------------------
# Defensive accessors — handle both old (flat) and new (nested) Qwen2-VL
# module layouts without needing to know the exact transformers version.
# ---------------------------------------------------------------------------

def _get_inner_model(hf_model: Qwen2VLForConditionalGeneration):
    """`Qwen2VLForConditionalGeneration.model` (Qwen2VLModel) — stable across versions."""
    return hf_model.model


def _get_text_decoder(hf_model: Qwen2VLForConditionalGeneration):
    """
    Returns the module holding `.layers` (ModuleList of decoder layers) and
    `.embed_tokens`. Newer transformers nests this under `.language_model`;
    older versions have `.layers`/`.embed_tokens` directly on `model.model`.
    """
    inner = _get_inner_model(hf_model)
    return getattr(inner, "language_model", inner)


def _get_visual_tower(hf_model: Qwen2VLForConditionalGeneration):
    inner = _get_inner_model(hf_model)
    visual = getattr(inner, "visual", None)
    if visual is None:
        visual = getattr(hf_model, "visual", None)  # last-resort fallback
    assert visual is not None, "Could not locate the Qwen2-VL vision tower (.visual)."
    return visual


def _get_text_hidden_size(config) -> int:
    text_cfg = getattr(config, "text_config", config)
    return text_cfg.hidden_size


def _get_text_vocab_size(config) -> int:
    text_cfg = getattr(config, "text_config", config)
    return text_cfg.vocab_size


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Qwen2VLXAttnForCausalLM(Qwen2VLForConditionalGeneration):
    """
    Drop-in replacement for Qwen2VLForConditionalGeneration that re-injects
    visual tokens via layered gated cross-attention instead of splicing them
    into the input embedding sequence.

    Usage:
        model = Qwen2VLXAttnForCausalLM.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
        )
        # cross_attn_modules are built automatically in __init__ from config.

    Training: freeze everything except `model.cross_attn_modules` (see
    train_qwenvl_xattn.py) — same Stage-1-only recipe as LLaVA's XAttn.
    """

    def __init__(self, config):
        super().__init__(config)

        self._visual_tokens: Optional[torch.Tensor] = None
        self._visual_kv_cache: dict = {}
        self._cross_attn_hooks: List = []

        text_dim = _get_text_hidden_size(config)
        # vision_dim == text_dim: Qwen2-VL's PatchMerger already projects
        # patches into the decoder's hidden size before they would have been
        # spliced into the input sequence.
        self.cross_attn_modules = nn.ModuleList()
        self._build_cross_attn_structure(text_dim=text_dim, vision_dim=text_dim)
        self._register_cross_attn_hooks()

    # ------------------------------------------------------------------
    # Cross-attention structure (mirrors llava_mistral_xattn.py's zones)
    # ------------------------------------------------------------------

    def _layer_index_map(self) -> dict:
        layers = _get_text_decoder(self).layers
        num_layers = len(layers)
        bottom_end = num_layers // 3
        middle_end = 2 * num_layers // 3

        layer_indices = []
        for i in range(bottom_end, middle_end, 2):   # middle: sparse, every 2
            layer_indices.append(i)
        for i in range(middle_end, num_layers):       # top: dense, every layer
            layer_indices.append(i)
        return {layer_idx: pos for pos, layer_idx in enumerate(layer_indices)}

    def _build_cross_attn_structure(
        self, text_dim: int, vision_dim: int,
        num_heads: int = 8, dropout: float = 0.1, sparse_topk: int = 64,
    ):
        layers = _get_text_decoder(self).layers
        num_layers = len(layers)
        bottom_end = num_layers // 3
        middle_end = 2 * num_layers // 3
        middle_set = set(range(bottom_end, middle_end, 2))

        self.cross_attn_modules = nn.ModuleList()
        for layer_idx in self._layer_index_map():
            ca = GatedCrossAttention(
                text_dim=text_dim,
                vision_dim=vision_dim,
                num_heads=num_heads,
                dropout=dropout,
                sparse_topk=sparse_topk if layer_idx in middle_set else None,
                num_patches=None,  # see module docstring: Qwen2-VL ViT already spatial-aware
            )
            self.cross_attn_modules.append(ca)

        n_total = len(layers)
        print(
            f"[Qwen2VL-XAttn] Built {len(self.cross_attn_modules)} cross-attention blocks "
            f"over {n_total} decoder layers (text_dim={text_dim}, vision_dim={vision_dim})\n"
            f"  Bottom (self-attn only): layers 0-{bottom_end - 1}\n"
            f"  Middle (sparse top-{sparse_topk}, every 2): "
            f"{list(range(bottom_end, middle_end, 2))}\n"
            f"  Top    (dense, every layer): {list(range(middle_end, n_total))}"
        )

    def _register_cross_attn_hooks(self):
        for h in self._cross_attn_hooks:
            h.remove()
        self._cross_attn_hooks = []

        layers = _get_text_decoder(self).layers
        layer_index_map = self._layer_index_map()
        for layer_idx, module_pos in layer_index_map.items():
            ca_module = self.cross_attn_modules[module_pos]
            hook = layers[layer_idx].register_forward_hook(
                self._make_cross_attn_hook(ca_module, module_pos)
            )
            self._cross_attn_hooks.append(hook)

    def rebuild_hooks_after_load(self):
        """Call after from_pretrained() loads a checkpoint with a different
        process/device — hooks are runtime-only and never serialized."""
        self._register_cross_attn_hooks()

    def _make_cross_attn_hook(self, ca_module: GatedCrossAttention, module_pos: int):
        model_ref = self

        def hook(module, layer_input: Tuple, layer_output):
            visual_tokens = model_ref._visual_tokens
            if visual_tokens is None:
                return layer_output  # no image this step -> skip

            cached_kv = model_ref._visual_kv_cache.get(module_pos)

            if isinstance(layer_output, torch.Tensor):
                hidden = layer_output
                is_tuple = False
            else:
                hidden = layer_output[0]
                is_tuple = True

            tgt_dtype = hidden.dtype
            vt = visual_tokens.to(dtype=tgt_dtype)
            ca_dtype = next(ca_module.parameters()).dtype
            out = ca_module(hidden.to(ca_dtype), vt.to(ca_dtype), precomputed_kv=cached_kv)
            out = out.to(tgt_dtype)

            return (out,) + layer_output[1:] if is_tuple else out

        return hook

    # ------------------------------------------------------------------
    # Visual token management (KV caching for generation)
    # ------------------------------------------------------------------

    def set_visual_tokens(self, visual_tokens: Optional[torch.Tensor]):
        self._visual_tokens = visual_tokens
        self._visual_kv_cache = {}
        if not self.training and visual_tokens is not None:
            for pos, ca_module in enumerate(self.cross_attn_modules):
                ca_dtype = next(ca_module.parameters()).dtype
                self._visual_kv_cache[pos] = ca_module.compute_visual_kv(
                    visual_tokens.to(dtype=ca_dtype)
                )

    def clear_visual_tokens(self):
        self._visual_tokens = None
        self._visual_kv_cache = {}

    # ------------------------------------------------------------------
    # Vision feature extraction (bypasses the original splice-into-sequence path)
    # ------------------------------------------------------------------

    def extract_visual_tokens(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.LongTensor
    ) -> torch.Tensor:
        """
        Returns (1, N_total_patches, text_hidden_size) visual tokens for a
        single image (batch_size=1 — matches every training script in this
        repo, which all use per-device batch size 1 + grad accumulation).

        If you need batch_size > 1 with multiple images, you must pad/split
        per-sample like llava_mistral_xattn.py's extract_visual_tokens does
        for variable-N images — not implemented here since it's unused by
        the current training recipe.
        """
        visual_tower = _get_visual_tower(self)
        pixel_values = pixel_values.type(visual_tower.dtype)
        vision_out = visual_tower(pixel_values, grid_thw=image_grid_thw)
        # Newer transformers wraps this in a ModelOutput with `.pooler_output`;
        # older versions return the embedding tensor directly.
        image_embeds = getattr(vision_out, "pooler_output", vision_out)
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        return image_embeds.unsqueeze(0)  # (1, N, D)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        """
        input_ids must be TEXT-ONLY (no <|image_pad|> placeholder tokens) —
        see dataset.py, which builds the chat template without an image
        content block. All visual grounding flows through cross-attention.
        """
        text_decoder = _get_text_decoder(self)
        inputs_embeds = text_decoder.embed_tokens(input_ids)

        if pixel_values is not None:
            self.clear_visual_tokens()
            visual_tokens = self.extract_visual_tokens(pixel_values, image_grid_thw)
            self.set_visual_tokens(visual_tokens.to(dtype=inputs_embeds.dtype))
        # else: keep whatever visual tokens were set on a previous decode step
        # (auto-regressive generation calls forward() once per new token).

        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device,
            )

        outputs = super().forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=None,           # bypass original splice-into-sequence path entirely
            image_grid_thw=None,
            labels=None,                 # compute loss ourselves — version-robust
            use_cache=use_cache,
            **kwargs,
        )

        loss = None
        if labels is not None:
            logits = outputs.logits
            vocab_size = _get_text_vocab_size(self.config)
            shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
            valid = (shift_labels != -100).sum()
            if valid == 0:
                loss = (shift_logits * 0.0).sum()
            else:
                loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(shift_logits, shift_labels)
            outputs.loss = loss

        return outputs

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        text_decoder = _get_text_decoder(self)
        inputs_embeds = text_decoder.embed_tokens(input_ids)

        if pixel_values is not None:
            self.clear_visual_tokens()
            visual_tokens = self.extract_visual_tokens(pixel_values, image_grid_thw)
            self.set_visual_tokens(visual_tokens.to(dtype=inputs_embeds.dtype))

        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device,
            )

        try:
            return super().generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pixel_values=None,
                image_grid_thw=None,
                **kwargs,
            )
        finally:
            self.clear_visual_tokens()
