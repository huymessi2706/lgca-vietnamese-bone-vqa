"""
LLaVA-Med + Layered Cross-Attention (LEARNED-GATE VARIANT)
=============================================================

This is a copy of llava_mistral_xattn.py with exactly one architectural
change: GatedCrossAttention (fixed alpha=0.05) is replaced by
GatedCrossAttentionLearnedGate (learned tanh gate, same zero-initialized
W^O). This is the experimental control for the paper's claim (Section
III-E2 / methods.tex) that a fixed residual coefficient avoids a
potential early-training optimization bottleneck that a learned gate,
combined with zero-initialized W^O, is exposed to.

Everything else — zone schedule (bottom/middle/top), sparse top-k,
positional embedding, hook mechanics, training/generation forward logic —
is unchanged, so differences in observed loss curves / final metrics are
attributable to the fixed-vs-learned-gate choice alone.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    MistralConfig,
    MistralModel,
    MistralForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..cross_attention_learned_gate import GatedCrossAttentionLearnedGate
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class LlavaMistralXAttnLearnedGateConfig(MistralConfig):
    model_type = "llava_mistral_xattn_learned_gate"

    def __init__(
        self,
        cross_attn_num_heads: int = 8,
        cross_attn_dropout: float = 0.1,
        cross_attn_sparse_topk: int = 64,   # ~11% of 576 CLIP patches; was 32
        cross_attn_num_patches: int = 576,  # CLIP-ViT-L/14@336 → 24×24 patches
        cross_attn_vision_dim: int = 0,     # set after Stage 1; enables checkpoint recovery
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cross_attn_num_heads   = cross_attn_num_heads
        self.cross_attn_dropout     = cross_attn_dropout
        self.cross_attn_sparse_topk = cross_attn_sparse_topk
        self.cross_attn_num_patches = cross_attn_num_patches
        # vision_dim is written by build_cross_attention_modules() so Stage 2/3 can
        # reconstruct the module structure before from_pretrained loads the weights.
        self.cross_attn_vision_dim  = cross_attn_vision_dim


# ---------------------------------------------------------------------------
# Model (decoder backbone + cross-attention injection)
# ---------------------------------------------------------------------------

class LlavaMistralXAttnLearnedGateModel(LlavaMetaModel, MistralModel):
    """
    MistralModel with GatedCrossAttentionLearnedGate injected at middle/top
    layers via PyTorch forward hooks. Visual tokens are stored in
    self._visual_tokens and consumed by the hooks during each forward pass.
    """

    config_class = LlavaMistralXAttnLearnedGateConfig

    def __init__(self, config: LlavaMistralXAttnLearnedGateConfig):
        super().__init__(config)
        self._visual_tokens: Optional[torch.Tensor] = None
        self._visual_kv_cache: dict = {}   # {module_pos: (K, V)} — filled at inference
        self._cross_attn_hooks: List = []
        self.cross_attn_modules = nn.ModuleList()

        # When loading a Stage 1+ checkpoint, from_pretrained calls __init__ first
        # then loads state_dict. Pre-building the module structure here ensures the
        # cross_attn_modules keys exist in the model before the state_dict is applied,
        # so the Stage 1 trained weights are loaded correctly instead of being skipped.
        if getattr(config, 'cross_attn_vision_dim', 0) > 0:
            self._init_cross_attn_structure(config.cross_attn_vision_dim)

    # ------------------------------------------------------------------
    # Cross-attention setup  (call once after vision tower is loaded)
    # ------------------------------------------------------------------

    def _layer_index_map(self) -> dict:
        """
        Returns the canonical mapping of decoder layer index → position in
        cross_attn_modules list.  Used both when building from scratch and
        when re-attaching hooks after a checkpoint load.
        """
        num_layers = len(self.layers)
        bottom_end = num_layers // 3
        middle_end = 2 * num_layers // 3

        layer_indices = []
        # Middle zone: sparse — every 2 layers
        for i in range(bottom_end, middle_end, 2):
            layer_indices.append(i)
        # Top zone: dense — every layer
        for i in range(middle_end, num_layers):
            layer_indices.append(i)

        return {layer_idx: pos for pos, layer_idx in enumerate(layer_indices)}

    def _register_hooks(self, layer_index_map: dict):
        """Attach forward hooks using the already-populated cross_attn_modules."""
        for h in self._cross_attn_hooks:
            h.remove()
        self._cross_attn_hooks.clear()

        for layer_idx, module_pos in layer_index_map.items():
            ca_module = self.cross_attn_modules[module_pos]
            hook = self.layers[layer_idx].register_forward_hook(
                self._make_cross_attn_hook(ca_module, module_pos)
            )
            self._cross_attn_hooks.append(hook)

    def _init_cross_attn_structure(self, vision_dim: int):
        """
        Build GatedCrossAttentionLearnedGate module list using dims from config.
        No hooks registered, no printing — safe to call from __init__().
        Called automatically when loading Stage 1+ checkpoints so that
        from_pretrained can populate the weights correctly.
        """
        num_layers  = len(self.layers)
        bottom_end  = num_layers // 3
        middle_end  = 2 * num_layers // 3
        num_heads   = getattr(self.config, "cross_attn_num_heads",   8)
        dropout     = getattr(self.config, "cross_attn_dropout",     0.1)
        sparse_topk = getattr(self.config, "cross_attn_sparse_topk", 64) or None
        num_patches = getattr(self.config, "cross_attn_num_patches", 576)
        text_dim    = self.config.hidden_size
        middle_set  = set(range(bottom_end, middle_end, 2))

        self.cross_attn_modules = nn.ModuleList()
        for layer_idx in self._layer_index_map():
            ca = GatedCrossAttentionLearnedGate(
                text_dim    = text_dim,
                vision_dim  = vision_dim,
                num_heads   = num_heads,
                dropout     = dropout,
                sparse_topk = sparse_topk if layer_idx in middle_set else None,
                num_patches = num_patches,
            )
            self.cross_attn_modules.append(ca)

    def build_cross_attention_modules(self, vision_dim: int, num_patches: int = 576):
        """
        Build GatedCrossAttentionLearnedGate modules from scratch and attach hooks.
        Call once after loading from the original HuggingFace model (Stage 1 fresh start).

        Zone boundaries (index-exclusive end):
          bottom : [0,            num_layers // 3)   — self-attention only
          middle : [num_layers//3, 2*num_layers//3)  — sparse cross-attn (every 2 layers)
          top    : [2*num_layers//3, num_layers)     — dense  cross-attn (every layer)
        """
        num_layers  = len(self.layers)
        bottom_end  = num_layers // 3
        middle_end  = 2 * num_layers // 3
        sparse_topk = getattr(self.config, "cross_attn_sparse_topk", 64) or None

        # Persist dims in config — this is what allows Stage 2/3 to reconstruct
        # the module structure before from_pretrained loads the checkpoint weights.
        self.config.cross_attn_num_patches = num_patches
        self.config.cross_attn_vision_dim  = vision_dim

        middle_layers = list(range(bottom_end, middle_end, 2))
        top_layers    = list(range(middle_end, num_layers))

        self._init_cross_attn_structure(vision_dim)
        self._register_hooks(self._layer_index_map())

        n_sparse = len(middle_layers)
        n_dense  = len(top_layers)
        print(
            f"[XAttn-LearnedGate] Built {len(self.cross_attn_modules)} cross-attention blocks  "
            f"(pos_embed={num_patches} patches, sparse_topk={sparse_topk})\n"
            f"  Bottom (self-attn only,  no xattn): layers 0–{bottom_end - 1}\n"
            f"  Middle (sparse top-{sparse_topk}, every 2): {middle_layers}  [{n_sparse} modules]\n"
            f"  Top    (dense, all tokens, every 1): {top_layers}  [{n_dense} modules]"
        )

    def rebuild_hooks(self):
        """
        Re-attach forward hooks after loading from a checkpoint.
        cross_attn_modules weights are already restored by from_pretrained —
        this only re-registers the runtime hooks (which are never serialized).
        """
        layer_index_map = self._layer_index_map()
        assert len(self.cross_attn_modules) == len(layer_index_map), (
            f"Checkpoint has {len(self.cross_attn_modules)} cross-attn modules "
            f"but architecture expects {len(layer_index_map)}. "
            "Model architecture may have changed since the checkpoint was saved."
        )
        self._register_hooks(layer_index_map)
        print(f"[XAttn-LearnedGate] Re-attached hooks for {len(layer_index_map)} cross-attention blocks (resumed from checkpoint)")

    def _make_cross_attn_hook(self, ca_module: GatedCrossAttentionLearnedGate, module_pos: int):
        """
        Returns a forward hook that applies cross-attention AFTER a decoder layer.

        During training : uses raw visual_tokens so gradients flow through k/v projections.
        During inference: uses precomputed KV cache from _visual_kv_cache[module_pos]
                          to skip redundant k/v projection on every decode step.
        """
        model_ref = self

        def hook(module, layer_input: Tuple, layer_output):
            visual_tokens = model_ref._visual_tokens
            if visual_tokens is None:
                return layer_output                      # no image → skip

            # Use cached K,V during inference; None during training (need gradients)
            cached_kv = model_ref._visual_kv_cache.get(module_pos)

            # Handle both tensor (new transformers) and tuple (old transformers) outputs.
            # Cast the cross-attn output back to the decoder layer's native dtype (FP16).
            # GatedCrossAttentionLearnedGate runs its LayerNorm in FP32 (autocast promotes
            # it), so the module output is FP32 even when the LLM layers are FP16.
            # Without the cast, each hook injects FP32 into the FP16 stream, causing
            # cumulative dtype drift across all hooked layers → logit explosion.
            if isinstance(layer_output, torch.Tensor):
                hidden = layer_output
                tgt_dtype = hidden.dtype
                # Cast visual_tokens to same dtype before cross-attn to avoid FP32 promotion
                vt = visual_tokens.to(dtype=tgt_dtype)
                _ca_dtype = next(ca_module.parameters()).dtype
                out = ca_module(hidden.to(_ca_dtype), vt.to(_ca_dtype), precomputed_kv=cached_kv)
                out = out.to(tgt_dtype)
                # DIAG: no_grad is mandatory — tensors created here must NOT enter the
                # autograd graph, otherwise the checkpoint recomputation sees a different
                # tensor count (forward: print runs; recomputation: _hook_diag_done=True
                # so print is skipped → CheckpointError 61 vs 56).
                if not getattr(model_ref, '_hook_diag_done', False) and module_pos == 0:
                    with torch.no_grad():
                        gate_val = torch.tanh(ca_module.gate).item()
                        print(f"  [Hook-0] hidden std={hidden.float().std().item():.4f} "
                              f"out std={out.float().std().item():.4f} "
                              f"gate(tanh)={gate_val:.4f}")
                return out
            else:
                hidden = layer_output[0]
                tgt_dtype = hidden.dtype
                vt = visual_tokens.to(dtype=tgt_dtype)
                _ca_dtype = next(ca_module.parameters()).dtype
                out = ca_module(hidden.to(_ca_dtype), vt.to(_ca_dtype), precomputed_kv=cached_kv)
                out = out.to(tgt_dtype)
                if not getattr(model_ref, '_hook_diag_done', False) and module_pos == 0:
                    with torch.no_grad():
                        gate_val = torch.tanh(ca_module.gate).item()
                        print(f"  [Hook-0] hidden std={hidden.float().std().item():.4f} "
                              f"out std={out.float().std().item():.4f} "
                              f"gate(tanh)={gate_val:.4f}")
                return (out,) + layer_output[1:]

        return hook

    # ------------------------------------------------------------------
    # Visual token management
    # ------------------------------------------------------------------

    def set_visual_tokens(self, visual_tokens: Optional[torch.Tensor]):
        """
        Store visual tokens and — during inference only — precompute K, V for
        every cross-attention module so each decode step skips redundant projection.

        During training self.training=True, so KV cache is NOT built: gradients
        must flow through k_proj/v_proj inside each forward call.
        """
        self._visual_tokens = visual_tokens
        self._visual_kv_cache = {}
        if not self.training and visual_tokens is not None:
            for pos, ca_module in enumerate(self.cross_attn_modules):
                ca_dtype = next(ca_module.parameters()).dtype
                self._visual_kv_cache[pos] = ca_module.compute_visual_kv(
                    visual_tokens.to(dtype=ca_dtype)
                )

    def clear_visual_tokens(self):
        """Call after generation is complete to free GPU memory."""
        self._visual_tokens = None
        self._visual_kv_cache = {}


# ---------------------------------------------------------------------------
# Causal LM head
# ---------------------------------------------------------------------------

class LlavaMistralXAttnLearnedGateForCausalLM(MistralForCausalLM, LlavaMetaForCausalLM):
    """
    Drop-in replacement for LlavaMistralForCausalLM that uses layered
    cross-attention (learned tanh gate variant) instead of visual prefix
    concatenation.
    """

    config_class = LlavaMistralXAttnLearnedGateConfig

    def __init__(self, config: LlavaMistralXAttnLearnedGateConfig):
        super(MistralForCausalLM, self).__init__(config)
        self.model   = LlavaMistralXAttnLearnedGateModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self) -> LlavaMistralXAttnLearnedGateModel:
        return self.model

    # ------------------------------------------------------------------
    # Visual token extraction (bypasses mm_projector)
    # ------------------------------------------------------------------

    def extract_visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run vision tower and return raw patch features.
        Shape: (B, N_patches, vision_hidden_size)

        NOTE: mm_projector is NOT applied here — the cross-attention modules
        have their own k_proj / v_proj to handle the vision → text dimension mapping.
        """
        vision_tower = self.get_vision_tower()
        if type(images) is list:
            # Variable-size images: encode each, pad to same N
            features = [vision_tower(img.unsqueeze(0)) for img in images]
            max_n = max(f.shape[1] for f in features)
            padded = [
                torch.nn.functional.pad(f, (0, 0, 0, max_n - f.shape[1]))
                for f in features
            ]
            return torch.cat(padded, dim=0)   # (B, max_N, D)
        else:
            return vision_tower(images)        # (B, N, D)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids:            Optional[torch.LongTensor]      = None,
        attention_mask:       Optional[torch.Tensor]           = None,
        position_ids:         Optional[torch.LongTensor]       = None,
        past_key_values:      Optional[List[torch.FloatTensor]] = None,
        inputs_embeds:        Optional[torch.FloatTensor]      = None,
        labels:               Optional[torch.LongTensor]       = None,
        use_cache:            Optional[bool]                   = None,
        output_attentions:    Optional[bool]                   = None,
        output_hidden_states: Optional[bool]                   = None,
        images:               Optional[torch.FloatTensor]      = None,
        image_sizes:          Optional[List[List[int]]]        = None,
        return_dict:          Optional[bool]                   = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            # --- Step 1: pure text embeddings ---
            # Replace IMAGE_TOKEN_INDEX with 0 (pad) so embed_tokens doesn't error;
            # visual context will come entirely from cross-attention.
            safe_ids = input_ids.clone()
            safe_ids[safe_ids < 0] = 0   # neutralise IMAGE_TOKEN_INDEX(-200), pad(-100), etc.
            inputs_embeds = self.get_model().embed_tokens(safe_ids)

            # --- Step 2: extract raw visual tokens for cross-attention ---
            if images is not None:
                # Clear stale tokens from the previous batch before setting new ones.
                # Clearing here (start of forward) instead of after super().forward()
                # ensures visual tokens remain available during gradient checkpointing
                # recomputation, which replays decoder layers inside the backward pass.
                self.model.clear_visual_tokens()
                visual_tokens = self.extract_visual_tokens(images)
                self.model.set_visual_tokens(
                    visual_tokens.to(dtype=inputs_embeds.dtype)
                )
            # If images is None (auto-regressive generation steps after the first),
            # _visual_tokens is already set from the first step — do nothing.

            # --- Step 3: build attention mask for text-only sequence ---
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2],
                    dtype=torch.bool,
                    device=inputs_embeds.device,
                )

        # --- DIAGNOSTICS: print tensor stats for first 3 forward steps ---
        # All ops here run under no_grad so they don't affect the autograd graph.
        _diag = getattr(self, '_diag_step', 0) < 3
        if _diag:
            with torch.no_grad():
                _s = getattr(self, '_diag_step', 0)
                print(f"\n[DIAG step={_s}]")
                ie = inputs_embeds.float()
                print(f"  inputs_embeds: dtype={inputs_embeds.dtype} shape={list(inputs_embeds.shape)} "
                      f"min={ie.min().item():.3f} max={ie.max().item():.3f} std={ie.std().item():.4f}")
                vt = self.model._visual_tokens
                if vt is not None:
                    vtf = vt.float()
                    print(f"  visual_tokens: dtype={vt.dtype} shape={list(vt.shape)} "
                          f"min={vtf.min().item():.3f} max={vtf.max().item():.3f} std={vtf.std().item():.4f}")
                else:
                    print("  visual_tokens: None (no image this step)")
                if labels is not None:
                    valid_count = (labels != IGNORE_INDEX).sum().item()
                    print(f"  labels: {valid_count}/{labels.numel()} non-masked tokens")
                    if valid_count == 0:
                        print("  WARNING: all labels are IGNORE_INDEX — loss will be NaN!")

        # Pass labels=None to the parent so we own loss computation.
        # transformers ≥ 4.46 changed MistralForCausalLM to use ForCausalLMLoss
        # (reduction='sum' / num_items_in_batch). When num_items_in_batch is not
        # forwarded from the Trainer it defaults to None and the function falls back
        # to dividing by (shift_labels != ignore_index).sum() — which should still
        # equal reduction='mean', BUT in some versions this path produces NaN when
        # all labels in a batch are masked. Computing the loss ourselves here is
        # version-agnostic and always correct.
        outputs = super(LlavaMistralXAttnLearnedGateForCausalLM, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if _diag:
            with torch.no_grad():
                lg = outputs.logits.float()
                print(f"  logits: dtype={outputs.logits.dtype} shape={list(outputs.logits.shape)} "
                      f"min={lg.min().item():.2f} max={lg.max().item():.2f} std={lg.std().item():.4f}")
            self.model._hook_diag_done = True   # suppress per-hook prints after step 0

        loss = None
        if labels is not None:
            logits = outputs.logits
            # Shift: token i predicts token i+1
            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
            valid_tokens = (shift_labels != IGNORE_INDEX).sum()
            if valid_tokens == 0:
                # Question filled the entire context window — answer was truncated away.
                # CE would be 0/0 = NaN. Return a differentiable zero instead so this
                # sample contributes nothing to the gradient without corrupting weights.
                loss = (shift_logits * 0.0).sum()
            else:
                loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(shift_logits, shift_labels)

        if _diag and loss is not None:
            print(f"  loss: {loss.item():.4e}")
            self._diag_step = getattr(self, '_diag_step', 0) + 1

        if return_dict is False:
            output = (outputs.logits,) + (outputs.past_key_values,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        inputs:      Optional[torch.Tensor] = None,
        images:      Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        position_ids   = kwargs.pop("position_ids",   None)
        attention_mask = kwargs.pop("attention_mask", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported in generate()")

        # Build text embeddings + store visual tokens before generation loop
        safe_ids = inputs.clone()
        safe_ids[safe_ids == IMAGE_TOKEN_INDEX] = 0
        inputs_embeds = self.get_model().embed_tokens(safe_ids)

        if images is not None:
            visual_tokens = self.extract_visual_tokens(images)
            self.model.set_visual_tokens(
                visual_tokens.to(dtype=inputs_embeds.dtype)
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.bool,
                device=inputs_embeds.device,
            )

        try:
            return super(MistralForCausalLM, self).generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        finally:
            # Always clear visual tokens after generation to free GPU memory
            self.model.clear_visual_tokens()

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images      = kwargs.pop("images",      None)
        image_sizes = kwargs.pop("image_sizes", None)
        # Call MistralForCausalLM directly (unbound) to skip the MRO gap.
        # super(MistralForCausalLM, self) would land on LlavaMetaForCausalLM which
        # has no prepare_inputs_for_generation, falling through to
        # GenerationMixin.prepare_inputs_for_generation which raises NotImplementedError.
        inputs = MistralForCausalLM.prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


# ---------------------------------------------------------------------------
# Register with HuggingFace AutoClass
# ---------------------------------------------------------------------------

AutoConfig.register("llava_mistral_xattn_learned_gate", LlavaMistralXAttnLearnedGateConfig)
AutoModelForCausalLM.register(LlavaMistralXAttnLearnedGateConfig, LlavaMistralXAttnLearnedGateForCausalLM)
