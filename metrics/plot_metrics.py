"""
Vẽ biểu đồ so sánh các model từ results/all_metrics.json.

Output (saved to results/figures/):
  fig1_bertscore_overview.png  — BERTScore-F1 tất cả model
  fig2_metrics_heatmap.png     — Heatmap tất cả metrics × model
  fig3_context_matrix.png      — 2×2 context transfer matrix
  fig4_rag_comparison.png      — So sánh các RAG strategy
  fig5_radar.png               — Radar chart các model chính
  fig6_lora_ablation.png       — 3-way ablation: Baseline vs LoRA-FT vs XAttn
  fig7_prediction_quality.png  — Prediction quality breakdown (empty/fragment/correct)

Usage:
    python plot_metrics.py
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

with open("results/all_metrics.json", encoding="utf-8") as f:
    data = json.load(f)

models = data["models"]

# Merge LoRA baseline metrics if available
_lora_path = "results/metrics/lora_baseline_metrics.json"
if os.path.exists(_lora_path):
    with open(_lora_path, encoding="utf-8") as f:
        _lora = json.load(f)
    _lm = _lora["models"].get("LoRA_FT", {})
    if _lm and "LoRA_FT" not in models:
        models["LoRA_FT"] = {
            "label":         "LoRA-FT\n(ablation)",
            "group":         "lora_ablation",
            "bleu1":         _lm.get("bleu1", 0.0),
            "bleu2":         _lm.get("bleu2", 0.0),
            "bleu4":         _lm.get("bleu4", 0.0),
            "rouge1":        _lm.get("rouge1", 0.0),
            "rouge2":        _lm.get("rouge2", 0.0),
            "rougeL":        _lm.get("rougeL", 0.0),
            "meteor":        _lm.get("meteor", 0.0),
            "bertscore_P":   _lm.get("bertscore_P", 0.0),
            "bertscore_R":   _lm.get("bertscore_R", 0.0),
            "bertscore_F1":  _lm.get("bertscore_F1", 0.0),
        }
        print(f"Merged LoRA-FT data from {_lora_path}")

GROUP_COLORS = {
    "baseline":         "#d62728",
    "lora_ablation":    "#e377c2",
    "xattn_full":       "#1f77b4",
    "xattn_minimal":    "#2ca02c",
    "context_transfer": "#9467bd",
    "rag_naive":        "#ff7f0e",
    "rag_aware":        "#8c564b",
}

GROUP_LABELS = {
    "baseline":         "Baseline (no FT)",
    "lora_ablation":    "LoRA Fine-tuning (ablation)",
    "xattn_full":       "XAttn – Full Context",
    "xattn_minimal":    "XAttn – Minimal Context",
    "context_transfer": "Context Transfer",
    "rag_naive":        "Naïve RAG",
    "rag_aware":        "RAG-Aware Training",
}

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — BERTScore-F1 overview (all models)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

names  = [v["label"] for v in models.values()]
scores = [v["bertscore_F1"] for v in models.values()]
colors = [GROUP_COLORS[v["group"]] for v in models.values()]

bars = ax.barh(names, scores, color=colors, edgecolor="white", height=0.65)
ax.axvline(x=0.9, color="red", linestyle="--", linewidth=1, alpha=0.5, label="0.90 threshold")
ax.set_xlabel("BERTScore-F1", fontsize=12)
ax.set_title("BERTScore-F1 Comparison — All Models", fontsize=14, fontweight="bold")
ax.set_xlim(0, 1.02)
ax.invert_yaxis()

for bar, score in zip(bars, scores):
    ax.text(score + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}", va="center", fontsize=8.5)

legend_patches = [mpatches.Patch(color=c, label=l) for g, (c, l) in
                  zip(GROUP_COLORS.keys(), [(v, GROUP_LABELS[k]) for k, v in GROUP_COLORS.items()])]
ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig1_bertscore_overview.png", dpi=150)
plt.close()
print("Saved fig1_bertscore_overview.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Metrics heatmap
# ─────────────────────────────────────────────────────────────────────────────
metric_keys  = ["bleu1", "bleu2", "bleu4", "rouge1", "rouge2", "rougeL", "meteor",
                "bertscore_P", "bertscore_R", "bertscore_F1"]
metric_labels = ["BLEU-1", "BLEU-2", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L",
                 "METEOR", "BERTScore-P", "BERTScore-R", "BERTScore-F1"]

short_labels = [v["label"].replace(" (", "\n(") for v in models.values()]
matrix = np.array([[v[m] for m in metric_keys] for v in models.values()])

fig, ax = plt.subplots(figsize=(14, 9))
im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

ax.set_xticks(range(len(metric_labels)))
ax.set_xticklabels(metric_labels, rotation=30, ha="right", fontsize=10)
ax.set_yticks(range(len(short_labels)))
ax.set_yticklabels(short_labels, fontsize=8.5)

for i in range(len(short_labels)):
    for j in range(len(metric_labels)):
        val = matrix[i, j]
        color = "black" if val < 0.6 else "white"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=7.5, color=color, fontweight="bold")

ax.set_title("All Metrics × All Models (green=high, red=low)", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig2_metrics_heatmap.png", dpi=150)
plt.close()
print("Saved fig2_metrics_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — 2×2 context transfer matrix (BERTScore-F1)
# ─────────────────────────────────────────────────────────────────────────────
matrix_2x2 = np.array([
    [0.9483, 0.8510],  # full-trained:    full-test, minimal-test
    [0.9041, 0.9162],  # minimal-trained: full-test, minimal-test
])
row_labels = ["Full-Context\nTraining", "Minimal-Context\nTraining"]
col_labels = ["Full-Context\nInference", "Minimal-Context\nInference"]

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(matrix_2x2, cmap="YlGn", vmin=0.83, vmax=0.96)

ax.set_xticks([0, 1]); ax.set_xticklabels(col_labels, fontsize=11)
ax.set_yticks([0, 1]); ax.set_yticklabels(row_labels, fontsize=11)
ax.set_xlabel("Inference Context", fontsize=12, fontweight="bold")
ax.set_ylabel("Training Context", fontsize=12, fontweight="bold")
ax.set_title("BERTScore-F1: Context Transfer Matrix\n(diagonal = matched train/test distribution)",
             fontsize=12, fontweight="bold")

for i in range(2):
    for j in range(2):
        is_diag = (i == j)
        ax.text(j, i, f"{matrix_2x2[i, j]:.4f}",
                ha="center", va="center", fontsize=16, fontweight="bold",
                color="darkgreen" if is_diag else "black")
        if is_diag:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                  edgecolor="darkgreen", linewidth=3)
            ax.add_patch(rect)

plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig3_context_matrix.png", dpi=150)
plt.close()
print("Saved fig3_context_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — RAG strategy comparison
# ─────────────────────────────────────────────────────────────────────────────
rag_models = {
    "No RAG\n(baseline)":          {"bleu1": 0.5094, "rougeL": 0.5763, "bertscore_F1": 0.9162},
    "BM25 Pre-RAG\n(naïve)":       {"bleu1": 0.2554, "rougeL": 0.3962, "bertscore_F1": 0.8763},
    "BM25 Post-RAG\n(naïve)":      {"bleu1": 0.2123, "rougeL": 0.3633, "bertscore_F1": 0.8658},
    "CLIP Pre-RAG\n(naïve)":       {"bleu1": 0.1941, "rougeL": 0.3456, "bertscore_F1": 0.8653},
    "BM25 Pre-RAG\n(RAG-aware)":   {"bleu1": 0.4311, "rougeL": 0.5398, "bertscore_F1": 0.9049},
    "No RAG infer\n(RAG-aware train)": {"bleu1": 0.3200, "rougeL": 0.3595, "bertscore_F1": 0.8750},
}

x     = np.arange(len(rag_models))
names = list(rag_models.keys())
width = 0.25

bleu1_vals  = [v["bleu1"]        for v in rag_models.values()]
rougel_vals = [v["rougeL"]       for v in rag_models.values()]
bert_vals   = [v["bertscore_F1"] for v in rag_models.values()]

fig, ax = plt.subplots(figsize=(14, 6))
b1 = ax.bar(x - width, bleu1_vals,  width, label="BLEU-1",        color="#4c72b0")
b2 = ax.bar(x,         rougel_vals, width, label="ROUGE-L",       color="#dd8452")
b3 = ax.bar(x + width, bert_vals,   width, label="BERTScore-F1",  color="#55a868")

ax.axhline(y=0.9162, color="#55a868", linestyle="--", linewidth=1.5, alpha=0.7,
           label="No-RAG BERTScore (0.9162)")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("RAG Strategy Comparison (Minimal Context — XAttn S1)", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4_rag_comparison.png", dpi=150)
plt.close()
print("Saved fig4_rag_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Radar chart (key models only)
# ─────────────────────────────────────────────────────────────────────────────
radar_models = {
    "Baseline":            [0.0,    0.0,    0.0,    0.0,    0.0,    0.0],
    "XAttn S1 (full)":     [0.6290, 0.5657, 0.7680, 0.7509, 0.7168, 0.9483],
    "XAttn S1 (minimal)":  [0.5094, 0.4022, 0.6141, 0.5763, 0.5394, 0.9162],
    "RAG-Aware Pre-RAG":   [0.4311, 0.3294, 0.5855, 0.5398, 0.4901, 0.9049],
    "BM25 Pre (naïve)":    [0.2554, 0.1614, 0.4656, 0.3962, 0.3063, 0.8763],
}
radar_metrics = ["BLEU-1", "BLEU-4", "ROUGE-1", "ROUGE-L", "METEOR", "BERTScore-F1"]

N = len(radar_metrics)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
radar_colors = ["#d62728", "#1f77b4", "#2ca02c", "#8c564b", "#ff7f0e"]

for (name, vals), color in zip(radar_models.items(), radar_colors):
    vals = vals + vals[:1]
    ax.plot(angles, vals, "o-", linewidth=2, color=color, label=name)
    ax.fill(angles, vals, alpha=0.06, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
ax.set_title("Radar Chart — Key Models Comparison", fontsize=13,
             fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig5_radar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig5_radar.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — LoRA Ablation: 3-way comparison (Baseline → LoRA-FT → XAttn)
# ─────────────────────────────────────────────────────────────────────────────
ablation_data = {
    "Zero-Shot\nBaseline":    {"bleu1": 0.0000, "rougeL": 0.0000, "meteor": 0.0000, "bertscore_F1": 0.0000},
    "LoRA-FT\n(r=16, 3ep)":  {"bleu1": 0.0000, "rougeL": 0.0119, "meteor": 0.0067, "bertscore_F1": 0.0514},
    "XAttn S1\n(minimal)":   {"bleu1": 0.5094, "rougeL": 0.5763, "meteor": 0.5394, "bertscore_F1": 0.9162},
}
abl_colors = ["#d62728", "#e377c2", "#2ca02c"]
metric_keys_abl  = ["bleu1", "rougeL", "meteor", "bertscore_F1"]
metric_labels_abl = ["BLEU-1", "ROUGE-L", "METEOR", "BERTScore-F1"]

x_abl = np.arange(len(ablation_data))
width_abl = 0.18

fig, ax = plt.subplots(figsize=(11, 6))

for i, (mk, ml) in enumerate(zip(metric_keys_abl, metric_labels_abl)):
    vals = [v[mk] for v in ablation_data.values()]
    offset = (i - 1.5) * width_abl
    bars = ax.bar(x_abl + offset, vals, width_abl, label=ml,
                  color=["#d62728", "#e377c2", "#2ca02c"][0] if i == 3 else None,
                  alpha=0.85 if i < 3 else 1.0)
    # Use distinct hatch patterns instead of changing color per group
    hatches = ["", "//", "xx", ".."]
    for bar, hatch in zip(bars, ["", "", ""]):
        bar.set_hatch(hatch)
    for bar, val in zip(bars, vals):
        if val > 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)

# Redraw with proper per-metric colors
ax.cla()
metric_colors_abl = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
for i, (mk, ml, mc) in enumerate(zip(metric_keys_abl, metric_labels_abl, metric_colors_abl)):
    vals = [v[mk] for v in ablation_data.values()]
    offset = (i - 1.5) * width_abl
    bars = ax.bar(x_abl + offset, vals, width_abl, label=ml, color=mc, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals):
        if val > 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)

ax.set_xticks(x_abl)
ax.set_xticklabels(list(ablation_data.keys()), fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0, 1.08)
ax.set_title("Ablation: Zero-Shot → LoRA Fine-tuning → Cross-Attention\n"
             "(same training data: 6807 samples, minimal context)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")

# Annotate the gap
ax.annotate("", xy=(2, 0.93), xytext=(1, 0.93),
            arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=2))
ax.text(1.5, 0.945, "+0.865 BERTScore\n(+94%)", ha="center", va="bottom",
        fontsize=9, color="darkgreen", fontweight="bold")

ax.annotate("", xy=(1, 0.12), xytext=(0, 0.12),
            arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
ax.text(0.5, 0.135, "+0.051", ha="center", va="bottom", fontsize=8.5, color="gray")

# Color x-tick labels to match group
for tick, color in zip(ax.get_xticklabels(), ["#d62728", "#e377c2", "#2ca02c"]):
    tick.set_color(color)
    tick.set_fontweight("bold")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig6_lora_ablation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig6_lora_ablation.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — Prediction quality breakdown (empty / hallucinated / correct)
# From manual analysis of lora_baseline_outputs_v2.json (first 30 samples)
# and xattn_minimal_s1_outputs.jsonl
# ─────────────────────────────────────────────────────────────────────────────
quality_data = {
    "Zero-Shot\nBaseline": {
        "correct":       0,   # all English → 0% against Vietnamese refs
        "partial":       0,
        "hallucinated": 100,  # generates English (domain/language mismatch)
        "empty":         0,
    },
    "LoRA-FT\n(r=16, 3ep)": {
        "correct":       0,   # 0/30 correct from manual check
        "partial":       4,   # ~17% non-empty but wrong (fragmented train data)
        "hallucinated": 13,   # fragments that are partially related
        "empty":        83,   # 25/30 EOS-degeneration
    },
    "XAttn S1\n(minimal)": {
        "correct":      54,   # 54% completely correct (from Section 8.7)
        "partial":      30,   # partial/paraphrase answers
        "hallucinated": 16,   # laterality errors + wrong diagnosis
        "empty":         0,   # no EOS-degeneration
    },
}

categories = ["correct", "partial", "hallucinated", "empty"]
cat_labels  = ["Correct / Near-correct", "Partial / Paraphrase", "Hallucinated / Wrong", "Empty (EOS-degeneration)"]
cat_colors  = ["#2ca02c", "#98df8a", "#d62728", "#7f7f7f"]

fig, ax = plt.subplots(figsize=(10, 5))
x_q = np.arange(len(quality_data))
bottoms = np.zeros(len(quality_data))

for cat, label, color in zip(categories, cat_labels, cat_colors):
    vals = [v[cat] for v in quality_data.values()]
    bars = ax.bar(x_q, vals, bottom=bottoms, label=label, color=color,
                  edgecolor="white", linewidth=0.8)
    for bar, val, bot in zip(bars, vals, bottoms):
        if val >= 6:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bot + val / 2,
                    f"{val}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
    bottoms += np.array(vals)

ax.set_xticks(x_q)
ax.set_xticklabels(list(quality_data.keys()), fontsize=12)
ax.set_ylabel("Percentage of Predictions (%)", fontsize=12)
ax.set_ylim(0, 108)
ax.set_title("Prediction Quality Breakdown by Model\n"
             "(manual analysis: 50 samples for XAttn, 30 for LoRA, estimated for Baseline)",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

# Annotate EOS degeneration
ax.annotate("EOS-degeneration:\n83% empty predictions",
            xy=(1, 83), xytext=(1.55, 65),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=9, color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

# Color x-tick labels
for tick, color in zip(ax.get_xticklabels(), ["#d62728", "#e377c2", "#2ca02c"]):
    tick.set_color(color)
    tick.set_fontweight("bold")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig7_prediction_quality.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig7_prediction_quality.png")

print(f"\nAll figures saved to {OUT_DIR}/")
