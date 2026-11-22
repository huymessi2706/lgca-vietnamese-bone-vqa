"""
GatedCrossAttention — text queries visual tokens.

Two modes:
  Dense  (sparse_topk=None) : every visual token is a key/value  [top layers]
  Sparse (sparse_topk=k)    : only the top-k most relevant visual tokens
                               (per query position) contribute    [middle layers]

Residual formula:
  T_out = T + sigmoid(gate) * CrossAttn(LayerNorm(T), pos_embed(V))

gate is initialized to -4 so sigmoid(-4) ≈ 0.018.

2D Positional Embedding:
  Learnable (H, W, vision_dim) grid added to visual tokens before K/V projection
  so the model knows spatial location of each patch (critical for medical imaging).
  Interpolated bilinearly if runtime resolution differs from the trained resolution.

KV Cache:
  compute_visual_kv() pre-projects visual tokens to K, V.
  Call it once before generation; pass the result as precomputed_kv to forward()
  to skip re-projection on every decode step.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2D Positional Embedding
# ---------------------------------------------------------------------------

class VisualPositionEmbedding(nn.Module):
    """
    Learnable 2D positional embedding for visual patch tokens.

    Initialized with truncated-normal (std=0.02), same as ViT position embeds.
    At runtime, if N != H*W (different image resolution), the embedding grid is
    interpolated bilinearly so the module is resolution-agnostic.

    Args:
        num_patches : expected number of patch tokens (H*W, must be a perfect square)
        vision_dim  : channel dimension of visual tokens
    """

    def __init__(self, num_patches: int, vision_dim: int):
        super().__init__()
        H = int(num_patches ** 0.5)
        assert H * H == num_patches, (
            f"num_patches ({num_patches}) must be a perfect square "
            f"(got sqrt ≈ {num_patches**0.5:.2f})"
        )
        self.H = self.W = H
        # (1, num_patches, vision_dim) — shared across batch
        self.embed = nn.Parameter(torch.zeros(1, num_patches, vision_dim))
        nn.init.trunc_normal_(self.embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, N, vision_dim)
        Returns:
            x + positional embedding, same shape
        """
        N = x.shape[1]
        if N == self.H * self.W:
            return x + self.embed
        # Resolution mismatch — interpolate the 2D grid
        pos = self.embed.reshape(1, self.H, self.W, -1).permute(0, 3, 1, 2)  # (1, D, H, W)
        H_new = W_new = int(N ** 0.5)
        pos = F.interpolate(pos, size=(H_new, W_new), mode='bilinear', align_corners=False)
        pos = pos.permute(0, 2, 3, 1).reshape(1, N, -1)                       # (1, N, D)
        return x + pos


# ---------------------------------------------------------------------------
# Gated Cross-Attention
# ---------------------------------------------------------------------------

class GatedCrossAttention(nn.Module):
    """
    Multi-head cross-attention where text hidden states are queries
    and visual tokens are keys/values.

    Features:
      - 2D learnable positional embedding on visual tokens (spatial awareness)
      - Sparse top-k masking for middle decoder layers
      - Scalar learned gate (init -4) to smoothly grow visual influence
      - compute_visual_kv() for KV caching during generation

    Args:
        text_dim    : hidden size of the LLM  (e.g. 4096 for Mistral-7B)
        vision_dim  : hidden size of the vision encoder (e.g. 1024 for CLIP-ViT-L)
        num_heads   : number of attention heads
        dropout     : attention dropout probability
        sparse_topk : attend only to the top-k visual tokens per query position.
                      None → dense mode (all N visual tokens).
        num_patches : number of visual patch tokens; if given, adds 2D pos embed.
                      None → no positional embedding.
    """

    def __init__(
        self,
        text_dim: int,
        vision_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        sparse_topk: Optional[int] = None,
        num_patches: Optional[int] = None,
    ):
        super().__init__()
        assert text_dim % num_heads == 0, (
            f"text_dim ({text_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.num_heads   = num_heads
        self.head_dim    = text_dim // num_heads
        self.scale       = math.sqrt(self.head_dim)
        self.sparse_topk = sparse_topk

        # 2D positional embedding for visual patches
        self.pos_embed = (
            VisualPositionEmbedding(num_patches, vision_dim)
            if num_patches is not None else None
        )

        # Q from text, K/V from vision
        self.q_proj   = nn.Linear(text_dim,   text_dim, bias=False)
        self.k_proj   = nn.Linear(vision_dim, text_dim, bias=False)
        self.v_proj   = nn.Linear(vision_dim, text_dim, bias=False)
        self.out_proj = nn.Linear(text_dim,   text_dim, bias=False)

        self.norm    = nn.LayerNorm(text_dim)
        self.dropout = nn.Dropout(dropout)

        # Fixed alpha=0.05: no learnable gate.
        # Learnable gates (sigmoid or tanh) stay near-zero because the gate gradient
        # ≈ ∂loss/∂out × cross_attn_out ≈ 0 when cross_attn is randomly initialized
        # (random cross_attn_out is uncorrelated with ∂loss/∂out → sum cancels).
        # Fixed alpha forces visual contribution so K/V/Q can learn; if visual info
        # is unhelpful the model drives out_proj weights → 0 to suppress it naturally.
        self.alpha = 0.05

        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight)
        # Zero-init out_proj: keeps initial cross_attn contribution = 0 so loss
        # starts at the same level as the frozen base LLM. Gradients still flow
        # through out_proj from the very first step (∂loss/∂out_proj ≠ 0 even at 0).
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    # ------------------------------------------------------------------
    # KV Cache helper (call once before generation loop)
    # ------------------------------------------------------------------

    def compute_visual_kv(
        self, visual_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pre-project visual tokens to multi-head K, V tensors.
        Call this once before the autoregressive generation loop and pass the
        result as `precomputed_kv` to forward() to avoid redundant projection
        on every decode step.

        Args:
            visual_tokens : (B, N, vision_dim)
        Returns:
            K : (B, H, N, head_dim)
            V : (B, H, N, head_dim)
        """
        if self.pos_embed is not None:
            visual_tokens = self.pos_embed(visual_tokens)
        B, N, _ = visual_tokens.shape
        H, d = self.num_heads, self.head_dim
        K = self.k_proj(visual_tokens).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        V = self.v_proj(visual_tokens).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        return K, V

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        text_hidden: torch.Tensor,                              # (B, L, text_dim)
        visual_tokens: Optional[torch.Tensor] = None,           # (B, N, vision_dim)
        precomputed_kv: Optional[Tuple[torch.Tensor,
                                       torch.Tensor]] = None,   # ((B,H,N,d), (B,H,N,d))
    ) -> torch.Tensor:                                          # (B, L, text_dim)
        """
        Either `visual_tokens` or `precomputed_kv` must be provided.
        During training: pass `visual_tokens` so gradients flow through k/v projections.
        During inference: pass `precomputed_kv` from compute_visual_kv() to skip
                          redundant re-projection on every decode step.
        """
        residual = text_hidden
        x = self.norm(text_hidden)

        B, L, D = x.shape
        H, d    = self.num_heads, self.head_dim

        # ── Queries from text ──
        Q = self.q_proj(x).view(B, L, H, d).transpose(1, 2)   # (B, H, L, d)

        # ── Keys / Values from vision ──
        if precomputed_kv is not None:
            K, V = precomputed_kv                               # (B, H, N, d)
            N = K.shape[2]
        else:
            assert visual_tokens is not None, \
                "Either visual_tokens or precomputed_kv must be provided"
            if self.pos_embed is not None:
                visual_tokens = self.pos_embed(visual_tokens)
            N = visual_tokens.shape[1]
            K = self.k_proj(visual_tokens).view(B, N, H, d).transpose(1, 2)
            V = self.v_proj(visual_tokens).view(B, N, H, d).transpose(1, 2)

        # ── Attention scores ──
        # Clamp before softmax: prevents FP16/BF16 overflow when Q or K have large
        # magnitudes early in training (which would give Inf attn scores → NaN output).
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, L, N)
        attn_scores = attn_scores.clamp(-10.0, 10.0)

        # ── Sparse top-k masking (middle layers) ──
        if self.sparse_topk is not None:
            k = min(self.sparse_topk, N)
            topk_vals, _ = attn_scores.topk(k, dim=-1)     # (B, H, L, k)
            threshold     = topk_vals[..., -1:]             # (B, H, L, 1)
            # Keep scores >= threshold; mask everything strictly below
            attn_scores = attn_scores.masked_fill(
                attn_scores < threshold, float('-inf')
            )

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                                 # (B, H, L, d)
        out = out.transpose(1, 2).contiguous().reshape(B, L, D)     # (B, L, D)
        out = self.out_proj(out)

        # ── Gated residual ──
        return residual + self.alpha * out
