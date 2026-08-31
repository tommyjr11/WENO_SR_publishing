#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPS="$ROOT/for_paper_results/tex_deps"
PAPER="$ROOT/WENO_NN"

export TEXINPUTS="$DEPS//:"
export BIBINPUTS="$PAPER//:"
export BSTINPUTS="$DEPS//:"

cd "$PAPER"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

if command -v gs >/dev/null 2>&1; then
  gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dPDFSETTINGS=/ebook \
    -dNOPAUSE -dQUIET -dBATCH \
    -sOutputFile=main_compressed.pdf main.pdf
  gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dPDFSETTINGS=/screen \
    -dNOPAUSE -dQUIET -dBATCH \
    -sOutputFile=main_small.pdf main.pdf
fi
