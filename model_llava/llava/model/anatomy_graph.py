"""
Anatomy-Constrained Graph (ACG) Module
=======================================

Refines visual tokens from BioMedCLIP BEFORE they enter GatedCrossAttention
by applying anatomical adjacency constraints between bone patches.

Key idea:
  - Each of the 576 CLIP patches is softly assigned to one of 14 bone regions
  - Visual self-attention is masked so patches only attend to anatomically
    adjacent regions → prevents anatomically impossible combinations
  - Output: refined visual tokens V_acg (same shape as input V)

Integration point in pipeline:
  BioMedCLIP → V (576×1024)
             → ACGModule → V_acg (576×1024)   ← only this changes
             → GatedCrossAttention (unchanged)

Bone region definitions (14 regions):
  0  Skull          1  Spine         2  Ribs
  3  Shoulder       4  Humerus       5  Elbow
  6  Radius/Ulna    7  Wrist         8  Hand
  9  Pelvis         10 Femur         11 Knee
  12 Tibia/Fibula   13 Ankle
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Bone anatomy adjacency (symmetric)
# ---------------------------------------------------------------------------

BONE_REGIONS = [
    "Skull", "Spine", "Ribs", "Shoulder",
    "Humerus", "Elbow", "Radius_Ulna", "Wrist", "Hand",
    "Pelvis", "Femur", "Knee", "Tibia_Fibula", "Ankle",
]
NUM_REGIONS = len(BONE_REGIONS)  # 14

# Edges: (region_a, region_b) — undirected
_EDGES = [
    (0, 1),   # Skull — Spine
    (1, 2),   # Spine — Ribs
    (1, 3),   # Spine — Shoulder
    (1, 9),   # Spine — Pelvis
    (2, 3),   # Ribs — Shoulder
    (3, 4),   # Shoulder — Humerus
    (4, 5),   # Humerus — Elbow
    (5, 6),   # Elbow — Radius/Ulna
    (6, 7),   # Radius/Ulna — Wrist
    (7, 8),   # Wrist — Hand
    (9, 10),  # Pelvis — Femur
    (10, 11), # Femur — Knee
    (11, 12), # Knee — Tibia/Fibula
    (12, 13), # Tibia/Fibula — Ankle
]


def build_adjacency_matrix() -> torch.Tensor:
    """
    Returns symmetric (14, 14) adjacency matrix with self-connections.
    adj[i, j] = 1.0 means regions i and j may attend to each other.
    """
    adj = torch.eye(NUM_REGIONS)
    for a, b in _EDGES:
        adj[a, b] = 1.0
        adj[b, a] = 1.0
    return adj


# ---------------------------------------------------------------------------
# BoneRegionDetector
# ---------------------------------------------------------------------------

class BoneRegionDetector(nn.Module):
    """
    Soft-assigns each visual patch to one of 14 bone regions.

    Input:  (B, N, vision_dim)
    Output: (B, N, num_regions)  — soft probabilities via softmax
    """

    def __init__(self, vision_dim: int = 1024, num_regions: int = NUM_REGIONS):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, vision_dim // 2),
            nn.GELU(),
            nn.Linear(vision_dim // 2, num_regions),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.proj(visual_tokens), dim=-1)


# ---------------------------------------------------------------------------
# AnatomyGraphAttention
# ---------------------------------------------------------------------------

class AnatomyGraphAttention(nn.Module):
    """
    Self-attention over visual patches, masked by anatomical adjacency.

    For each pair of patches (i, j):
      - Compute soft mask: mask[i,j] = sum_{r,s} P[i,r] * adj[r,s] * P[j,s]
        where P[i,r] is the region probability of patch i
      - mask[i,j] ≈ 1 when patches i and j belong to adjacent regions
      - mask[i,j] ≈ 0 when patches are anatomically unrelated
      → Prevents cross-region attention between non-adjacent bone areas

    Input:
      visual_tokens  : (B, N, vision_dim)
      region_probs   : (B, N, num_regions) from BoneRegionDetector
    Output:
      refined_tokens : (B, N, vision_dim)
    """

    def __init__(
        self,
        vision_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert vision_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = vision_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)

        self.q_proj  = nn.Linear(vision_dim, vision_dim, bias=False)
        self.k_proj  = nn.Linear(vision_dim, vision_dim, bias=False)
        self.v_proj  = nn.Linear(vision_dim, vision_dim, bias=False)
        self.out_proj = nn.Linear(vision_dim, vision_dim, bias=False)
        self.norm     = nn.LayerNorm(vision_dim)
        self.dropout  = nn.Dropout(dropout)

        # Fixed adjacency — not trained, just a structural prior
        self.register_buffer("adjacency", build_adjacency_matrix())  # (14, 14)

        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight)
        nn.init.zeros_(self.out_proj.weight)  # zero-init for stable residual

    def _build_anatomy_mask(
        self,
        region_probs: torch.Tensor,  # (B, N, R)
    ) -> torch.Tensor:               # (B, N, N)  values in [0,1]
        """
        Soft anatomy mask via:
          mask = region_probs @ adjacency @ region_probs^T
        mask[b, i, j] ∈ [0,1]: how anatomically compatible patches i and j are.
        """
        B, N, R = region_probs.shape
        # (B, N, R) @ (R, R) → (B, N, R)
        adj_expanded = self.adjacency.unsqueeze(0).expand(B, -1, -1)  # (B, R, R)
        weighted = torch.bmm(region_probs, adj_expanded)               # (B, N, R)
        # (B, N, R) @ (B, R, N) → (B, N, N)
        mask = torch.bmm(weighted, region_probs.transpose(1, 2))       # (B, N, N)
        return mask  # values in [0, R] — will be used as multiplicative weight

    def forward(
        self,
        visual_tokens: torch.Tensor,   # (B, N, D)
        region_probs: torch.Tensor,    # (B, N, R)
    ) -> torch.Tensor:                 # (B, N, D)
        residual = visual_tokens
        x = self.norm(visual_tokens)
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim

        Q = self.q_proj(x).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        K = self.k_proj(x).view(B, N, H, d).transpose(1, 2)
        V = self.v_proj(x).view(B, N, H, d).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, N, N)
        scores = scores.clamp(-10.0, 10.0)

        # Anatomy mask: low values → soft penalty (log domain)
        # mask values near 0 → penalise, near 1 → allow
        anatomy_mask = self._build_anatomy_mask(region_probs)          # (B, N, N)
        anatomy_mask = anatomy_mask.unsqueeze(1).expand_as(scores)     # (B, H, N, N)
        # Add large negative bias where mask is near 0
        scores = scores + torch.log(anatomy_mask.clamp(min=1e-6))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                           # (B, H, N, d)
        out = out.transpose(1, 2).contiguous().reshape(B, N, D)
        out = self.out_proj(out)

        return residual + out


# ---------------------------------------------------------------------------
# ACGModule — drop-in visual token refiner
# ---------------------------------------------------------------------------

class ACGModule(nn.Module):
    """
    Full ACG pipeline: detect regions → anatomy-constrained self-attention
    → residual blend back into original visual tokens.

    Usage:
        acg = ACGModule(vision_dim=1024)
        V_refined = acg(V)   # same shape as V

    alpha=0.05 matches the GatedCrossAttention convention:
    small initial contribution, model drives it higher if useful.
    """

    def __init__(
        self,
        vision_dim: int = 1024,
        num_regions: int = NUM_REGIONS,
        num_heads: int = 8,
        dropout: float = 0.1,
        alpha: float = 0.05,
    ):
        super().__init__()
        self.region_detector   = BoneRegionDetector(vision_dim, num_regions)
        self.graph_attention   = AnatomyGraphAttention(vision_dim, num_heads, dropout)
        self.alpha             = alpha

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_tokens : (B, N, vision_dim)  — raw BioMedCLIP patch features
        Returns:
            refined       : (B, N, vision_dim)  — anatomy-constrained features
        """
        region_probs = self.region_detector(visual_tokens)          # (B, N, 14)
        refined      = self.graph_attention(visual_tokens, region_probs)  # (B, N, D)
        # alpha blending: residual already inside graph_attention.out_proj (zero-init)
        # but we additionally scale the full module output
        return visual_tokens + self.alpha * (refined - visual_tokens)

    def extra_repr(self) -> str:
        return (
            f"vision_dim={self.graph_attention.q_proj.in_features}, "
            f"num_regions={NUM_REGIONS}, alpha={self.alpha}"
        )
