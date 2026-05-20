"""
Generates all Chapter 3 diagrams as PDFs to replace inline TikZ.
Run: python3 generate_chapter3_diagrams.py
Output: figures/arch.pdf, figures/fusion.pdf, figures/workflow.pdf, figures/userflow.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

OUT = "figures"

# ── shared helpers ─────────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, text, fc, ec="black", fontsize=9, lw=1.2, rad=0.04):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad={rad}", fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            multialignment="center", zorder=4)

def diamond(ax, x, y, w, h, text, fc, ec="black", fontsize=8.5):
    xs = [x, x + w/2, x, x - w/2, x]
    ys = [y + h/2, y, y - h/2, y, y + h/2]
    ax.fill(xs, ys, fc=fc, ec=ec, lw=1.2, zorder=3)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            multialignment="center", zorder=4)

def arrow(ax, x1, y1, x2, y2, label="", lpos="top", color="black", lsize=8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3), zorder=3)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        dy = 0.06 if lpos == "top" else -0.1
        dx = 0.06 if lpos == "right" else (-0.06 if lpos == "left" else 0)
        ax.text(mx+dx, my+dy, label, ha="center", va="center",
                fontsize=lsize, color="dimgray")

def hline(ax, x1, x2, y, color="black", lw=1.3):
    ax.plot([x1, x2], [y, y], color=color, lw=lw, zorder=3)

def vline(ax, x, y1, y2, color="black", lw=1.3):
    ax.plot([x, x], [y1, y2], color=color, lw=lw, zorder=3)


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3.1 — System Architecture (4-layer)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10); ax.set_ylim(0, 9); ax.axis("off")

LAYERS = [
    (8.5, 8.2, "#e8d5f5", "Presentation Layer"),
    (8.5, 6.2, "#d0e4f7", "API Layer"),
    (8.5, 4.2, "#fde8d0", "Detection Pipeline"),
    (8.5, 2.1, "#d5f0d5", "Model Layer"),
]
for lx, ly, lc, lt in LAYERS:
    ax.text(lx, ly, lt, ha="center", va="center", fontsize=8.5,
            color="gray", style="italic", fontweight="bold")

# Layer bands
for y_top, y_bot, fc in [(8.8, 7.4, "#e8d5f5"), (7.3, 5.5, "#d0e4f7"),
                           (5.4, 3.2, "#fde8d0"), (3.1, 0.8, "#d5f0d5")]:
    ax.fill_betweenx([y_bot, y_top], 0.1, 7.9, alpha=0.18, color=fc)

# Presentation layer
rbox(ax, 2.0, 8.2, 2.8, 0.75, "React Frontend\nPort 3000", "#e8d5f5")
rbox(ax, 5.5, 8.2, 2.8, 0.75, "Upload Page\nResults Page", "#f0e8fa")

# API layer
rbox(ax, 2.0, 6.2, 2.8, 0.75, "FastAPI Backend\nPort 8000", "#d0e4f7")
rbox(ax, 5.5, 6.2, 2.8, 0.75, "/detect\n/heatmap", "#e4f0fa")

# Detection pipeline
rbox(ax, 1.2, 4.3, 2.0, 0.75, "Pre-\nprocessing", "#fde8d0")
rbox(ax, 3.8, 4.3, 2.0, 0.75, "Parallel\nInference", "#fde8d0")
rbox(ax, 6.4, 4.3, 2.0, 0.75, "Confidence\nFusion", "#fde8d0")
arrow(ax, 2.2, 4.3, 2.8, 4.3)
arrow(ax, 4.8, 4.3, 5.4, 4.3)

# Model layer
rbox(ax, 1.2, 1.9, 2.0, 0.75, "ViT\nw = 0.50", "#d5f0d5")
rbox(ax, 3.8, 1.9, 2.0, 0.75, "F3Net\nw = 0.35", "#d5f0d5")
rbox(ax, 6.4, 1.9, 2.0, 0.75, "EfficientNet\nw = 0.15", "#d5f0d5")

# Result cache
rbox(ax, 3.8, 0.5, 2.8, 0.6, "Result Cache & Heatmaps", "#eeeeee")

# Inter-layer arrows
arrow(ax, 2.0, 7.83, 2.0, 6.58)
arrow(ax, 2.0, 5.83, 3.8, 4.68, label="dispatch", lpos="top")
arrow(ax, 3.8, 3.92, 1.2, 2.28)
arrow(ax, 3.8, 3.92, 3.8, 2.28)
arrow(ax, 3.8, 3.92, 6.4, 2.28)
arrow(ax, 6.4, 3.92, 6.4, 3.28)
arrow(ax, 6.4, 3.28, 3.8, 0.8)

plt.tight_layout()
plt.savefig(f"{OUT}/arch.pdf", bbox_inches="tight")
plt.savefig(f"{OUT}/arch.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved arch")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3.3 — Confidence Fusion Example
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 4))
ax.set_xlim(0, 9); ax.set_ylim(0, 4); ax.axis("off")

rbox(ax, 1.5, 3.2, 2.4, 0.8, "ViT Output\nfake_prob = 0.85", "#d0eef7", fontsize=9)
rbox(ax, 4.5, 3.2, 2.4, 0.8, "F3Net Output\nfake_prob = 0.92", "#d0eef7", fontsize=9)
rbox(ax, 7.5, 3.2, 2.4, 0.8, "EfficientNet\nfake_prob = 0.78", "#d0eef7", fontsize=9)

rbox(ax, 4.5, 1.9, 8.4, 0.85,
     "0.50 × 0.85  +  0.35 × 0.92  +  0.15 × 0.78  =  0.859",
     "#fde8c0", fontsize=10)

rbox(ax, 4.5, 0.65, 4.0, 0.75,
     "Final Verdict:  Fake  (85.9 %)", "#ffc0b0", fontsize=10)

# arrows from models to fusion
for mx, lbl in [(1.5, "w = 0.50"), (4.5, "w = 0.35"), (7.5, "w = 0.15")]:
    arrow(ax, mx, 2.80, mx, 2.33)
    ax.text(mx + 0.15, 2.57, lbl, fontsize=8, color="dimgray")

arrow(ax, 4.5, 1.48, 4.5, 1.03)

plt.tight_layout()
plt.savefig(f"{OUT}/fusion.pdf", bbox_inches="tight")
plt.savefig(f"{OUT}/fusion.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved fusion")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3.2 — Unified Detection Workflow
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 13))
ax.set_xlim(-1, 11); ax.set_ylim(0, 14); ax.axis("off")

BW = 3.6   # box width
BH = 0.75  # box height

# Shared entry nodes
rbox(ax, 5, 13.3, BW, BH, "User Uploads Media", "#e0e0e0")
rbox(ax, 5, 11.9, BW, BH, "Validate File (Type & Size)", "#b2ebf2")
diamond(ax, 5, 10.6, 2.6, 0.8, "Valid?", "#fff9c4")
rbox(ax, 8.3, 10.6, 2.0, BH, "Return Error", "#ffcdd2", fontsize=8.5)
diamond(ax, 5, 9.2, 2.8, 0.85, "Image or\nVideo?", "#fff9c4")

# Labels above branches
ax.text(1.8, 8.6, "Image Path", fontsize=8.5, color="#1565c0", fontweight="bold")
ax.text(7.2, 8.6, "Video Path", fontsize=8.5, color="#00695c", fontweight="bold")

# Image path (left, x=2.0)
rbox(ax, 2.0, 7.9, 3.2, BH, "Preprocess (OpenCV)", "#bbdefb")
rbox(ax, 2.0, 6.6, 3.2, BH, "Parallel Inference:\nViT + F3Net + EfficientNet", "#bbdefb", fontsize=8.5)
rbox(ax, 2.0, 5.2, 3.2, BH, "Confidence Fusion\n(Weighted Average)", "#bbdefb", fontsize=8.5)

# Video path (right, x=8.0)
rbox(ax, 8.0, 7.9, 3.2, BH, "Extract Keyframes\n(1 sec intervals)", "#b2dfdb", fontsize=8.5)
rbox(ax, 8.0, 6.6, 3.2, BH, "Per-frame Detection\n(Image Pipeline)", "#b2dfdb", fontsize=8.5)
rbox(ax, 8.0, 5.2, 3.2, BH, "Temporal Consistency\n+ Aggregation", "#b2dfdb", fontsize=8.5)

# Merge
rbox(ax, 5, 3.8, 4.0, BH, "Generate Heatmaps & Explanations", "#b2ebf2")
rbox(ax, 5, 2.4, 3.6, BH, "Return JSON Response", "#e0e0e0")

# ── entry arrows
arrow(ax, 5, 12.93, 5, 12.28)
arrow(ax, 5, 11.53, 5, 11.0)
arrow(ax, 5, 10.2, 8.3, 10.6, label="No", lpos="top")   # No branch (horizontal)
arrow(ax, 5, 10.2, 5, 9.63, label="Yes", lpos="right")   # Yes branch

# ── branch from diamond to image/video paths (L-shapes)
# Image branch: type.west  → left → down → Image pp.north
# type left point is at (5 - 1.4, 9.2) = (3.6, 9.2) ... actually diamond half-width=1.4
# Let's route: from (3.6, 9.2) go left to x=2.0, then down to pp.north y=8.28
vline(ax, 2.0, 8.28, 9.2)
hline(ax, 2.0, 3.6, 9.2)
ax.annotate("", xy=(2.0, 8.28), xytext=(2.0, 8.3),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3), zorder=3)
ax.text(2.6, 9.32, "Image", fontsize=8, color="dimgray")

# Video branch: from (6.4, 9.2) go right to x=8.0, then down to kf.north y=8.28
vline(ax, 8.0, 8.28, 9.2)
hline(ax, 6.4, 8.0, 9.2)
ax.annotate("", xy=(8.0, 8.28), xytext=(8.0, 8.3),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3), zorder=3)
ax.text(7.0, 9.32, "Video", fontsize=8, color="dimgray")

# Image path chain
arrow(ax, 2.0, 7.53, 2.0, 6.98)
arrow(ax, 2.0, 6.22, 2.0, 5.58)

# Video path chain
arrow(ax, 8.0, 7.53, 8.0, 6.98)
arrow(ax, 8.0, 6.22, 8.0, 5.58)

# T-junction convergence to heat
JUNC_Y = 4.18
vline(ax, 2.0, 4.84, JUNC_Y)   # fus.south → junction y
vline(ax, 8.0, 4.84, JUNC_Y)   # tc.south  → junction y
hline(ax, 2.0, 8.0, JUNC_Y)    # horizontal bar
arrow(ax, 5, JUNC_Y, 5, 4.18)  # single arrow down to heat.north

arrow(ax, 5, 3.43, 5, 2.78)

plt.tight_layout()
plt.savefig(f"{OUT}/workflow.pdf", bbox_inches="tight")
plt.savefig(f"{OUT}/workflow.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved workflow")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3.4 — User Interaction Flow
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

# Row shading
for y0, y1, fc, lbl, lc in [(3.0, 4.8, "#f3e5f5", "Frontend",  "#7b1fa2"),
                              (1.8, 3.0, "#e3f2fd", "Backend",   "#1565c0"),
                              (0.4, 1.8, "#e8f5e9", "Results",   "#2e7d32")]:
    ax.fill_betweenx([y0, y1], 0.2, 9.8, alpha=0.25, color=fc)
    ax.text(0.05, (y0+y1)/2, lbl, fontsize=9, color=lc, fontweight="bold",
            va="center", ha="left", rotation=90)

# Frontend row
rbox(ax, 2.2, 4.0, 2.6, 0.75, "Upload Image\n(Drag & Drop)", "#e1bee7", fontsize=9)
rbox(ax, 5.0, 4.0, 2.6, 0.75, "Preview &\nValidate", "#e1bee7", fontsize=9)
rbox(ax, 7.8, 4.0, 2.6, 0.75, "Click Analyse", "#e1bee7", fontsize=9)

# Backend row
rbox(ax, 4.0, 2.35, 2.6, 0.75, "POST /detect\nFastAPI", "#bbdefb", fontsize=9)
rbox(ax, 7.2, 2.35, 2.6, 0.75, "3-Model\nInference", "#bbdefb", fontsize=9)

# Results row
rbox(ax, 2.0, 1.1, 2.4, 0.75, "Verdict\nBanner", "#c8e6c9", fontsize=9)
rbox(ax, 5.0, 1.1, 2.4, 0.75, "Heatmap\nViewer", "#c8e6c9", fontsize=9)
rbox(ax, 8.0, 1.1, 2.6, 0.75, "Explanations\n& Vote Table", "#c8e6c9", fontsize=9)

# Frontend chain
arrow(ax, 3.5, 4.0, 3.7, 4.0)
arrow(ax, 6.3, 4.0, 6.5, 4.0)

# Frontend → backend
arrow(ax, 7.8, 3.62, 6.0, 2.72)
arrow(ax, 5.3, 2.35, 5.9, 2.35)

# Backend → results
arrow(ax, 7.2, 1.98, 8.0, 1.49)

# Results chain (right to left)
arrow(ax, 6.7, 1.1, 6.2, 1.1)
arrow(ax, 3.8, 1.1, 3.2, 1.1)

plt.tight_layout()
plt.savefig(f"{OUT}/userflow.pdf", bbox_inches="tight")
plt.savefig(f"{OUT}/userflow.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved userflow")

print("\nAll Chapter 3 diagram PDFs saved to figures/")
