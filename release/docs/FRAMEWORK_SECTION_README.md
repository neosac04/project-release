# Framework and System Design Section - Integration Guide

## Files Created

1. **`project_report_framework_section.tex`** — Complete chapter with all text, tables, and figure references
2. **`tikz_diagrams.tex`** — TikZ code for all 5 diagrams (vector graphics, renders natively in LaTeX)

## How to Use

### Option 1: Direct Integration (Recommended)

Add these lines to your main LaTeX document (e.g., `main.tex`):

```latex
% In your main document preamble (with other \usepackage commands):
\usepackage{tikz}
\usetikzlibrary{shapes, arrows, positioning, calc}

% In your document where you want the chapter:
\input{tikz_diagrams.tex}      % Must come BEFORE the chapter
\input{project_report_framework_section.tex}
```

### Option 2: Copy-Paste

If you prefer to copy content directly:

1. Copy all content from `project_report_framework_section.tex` into your main document
2. Copy the TikZ diagram code from `tikz_diagrams.tex` into a `figures/` directory as separate files, or paste directly into your chapter

## Required LaTeX Packages

The diagrams use the following TikZ libraries (add to your preamble):

```latex
\usepackage{tikz}
\usetikzlibrary{shapes, arrows, positioning, calc}
\usepackage{amsmath}    % For \times symbol in equations
```

## Figure Compilation

When you compile your LaTeX document, the TikZ code will automatically generate the diagrams as vector graphics. First compile may take longer (10-20 sec) as TikZ renders the diagrams. Subsequent compiles are cached.

## Customization

### Colors
To change diagram colors globally, modify the TikZ styles at the top of each diagram:
- `fill=blue!20` means 20% opacity blue fill
- Change to `fill=red!20`, `fill=green!20`, etc. to customize

### Text Size
To make text larger/smaller, use `\footnotesize`, `\small`, `\normalsize`, `\large`, etc. within nodes:
```latex
\node[...] (label) at (...) {\small Smaller text};
```

### Figure Size
Wrap the `tikzpicture` environment in a `scale` command:
```latex
\begin{tikzpicture}[scale=0.8]  % 0.8 = 80% of original size
```

## Diagram Details

| Figure | Description | References |
|--------|-------------|-----------|
| Fig 4.1 | System Architecture (4 layers) | Models, fusion weights |
| Fig 4.2 | Image Detection Pipeline | Processing steps, timing |
| Fig 4.3 | Video Detection (optional) | Keyframe sampling, temporal analysis |
| Fig 4.4 | User Interaction Flow | 7-step journey from upload to results |
| Fig 4.5 | Ensemble Fusion Example | Weighted averaging with concrete numbers |

## Matching Your Report Style

The text in `project_report_framework_section.tex` is written to match academic technical writing:

- **Conversational but precise** — explains *why* choices were made, not just *what* they are
- **Self-contained** — sections can be read independently
- **Referenced throughout** — uses `\ref{c4:fig1}`, `\ref{c4:tab1}` for cross-references
- **Technical depth** — includes equations, component specs, and architectural rationale

If your report uses different formatting (e.g., different chapter numbering, color scheme), you can:

1. Replace `\chapter{FRAMEWORK AND SYSTEM DESIGN}` with your section format
2. Update figure/table references from `c4:fig1` to match your numbering scheme (e.g., `c6:fig1` for Chapter 6)
3. Adjust font sizes in `\node` commands if needed

## What's Included

✅ Architecture overview (4-layer stack)
✅ Model specifications (ViT, F3Net, EfficientNet with AUC/weights)
✅ Fusion strategy (weighted averaging, Platt scaling, calibration)
✅ Image detection workflow (8-step pipeline)
✅ Video detection extension (keyframe sampling approach)
✅ Frontend/backend components breakdown
✅ API endpoint reference table
✅ Tech stack summary table
✅ User journey flow (7 steps)
✅ 5 supporting diagrams in TikZ

## What's NOT Included

❌ Reverse image search / TinEye API (as requested)
❌ Training scripts (already in `backend/train_*.py`)
❌ Calibration details (already in `backend/calibrate_models.py`)

## Tips for Final Report

1. **Proofread diagrams** — They render correctly in LaTeX, but check spacing/alignment in your final PDF
2. **Check page breaks** — Long diagrams (especially Fig 4.2) may need landscape orientation:
   ```latex
   \begin{landscape}
     \input{tikz_diagrams.tex}
   \end{landscape}
   ```
3. **Cross-reference integrity** — If you change figure labels, update all `\ref{c4:figX}` in the text
4. **Compile warnings** — TikZ is verbose; ignore most warnings unless they mention "undefined control sequence"

## Questions?

Refer to the inline comments in `tikz_diagrams.tex` for diagram-specific configuration notes.
