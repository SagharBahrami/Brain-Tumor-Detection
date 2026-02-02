# MSc Thesis: Explainable Deep Learning for Brain Tumor Detection

LaTeX source files for the Master of Science thesis submitted to California State University San Marcos.

**Author**: Saghar Bahrami
**Advisor**: Dr. Sreedevi Gutta
**Program**: Master of Science in Computer Science
**Date**: Spring 2026

## Thesis Topic

Slice-level binary tumor detection from brain MRI scans using deep learning with Grad-CAM explainability. Compares CNN architectures (ResNet-50, EfficientNet-B2) with Vision Transformers (DeiT-Base) on the BraTS 2021 dataset.

## Directory Structure

```
thesis_draft/
├── main.tex              # Main LaTeX document
├── main.pdf              # Compiled thesis (74 pages)
├── references.bib        # Bibliography
├── chapters/
│   ├── abstract.tex
│   ├── acknowledgments.tex
│   ├── 01_introduction.tex
│   ├── 02_literature_review.tex
│   ├── 03_methodology.tex
│   ├── 04_results.tex
│   ├── 05_discussion.tex
│   └── 06_conclusion.tex
└── figures/              # Thesis figures
```

## Compilation

```bash
cd thesis_draft
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or using latexmk:
```bash
latexmk -pdf main.tex
```

## Key Results

| Model | Test Accuracy | Test F1 |
|-------|---------------|---------|
| ResNet-50 | 94.69% | 93.72% |
| EfficientNet-B2 | 93.67% | 92.70% |
| DeiT-Base | 93.46% | 92.35% |

**Recommendation**: EfficientNet-B2 for clinical deployment (best explainability despite slightly lower accuracy).