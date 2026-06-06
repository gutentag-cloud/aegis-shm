#!/usr/bin/env bash
# Reproduce every Aegis deliverable from scratch.
set -e
cd "$(dirname "$0")"

echo "[1/5] Verifying the physics engines ........................"
python3 src/beam_simulation.py     # FEM vs. analytical solution
python3 src/check_solver.py        # differentiable solver vs. FEM

echo "[2/5] Running the full experiment suite (~5 min) .........."
python3 src/run_experiments.py     # -> results/*.json, website/data/results.js, models/

echo "[3/6] Generating figures, table, macros, social card ......"
python3 src/make_figures.py        # -> paper/figures/*, paper/table_results.tex, results_macros.tex
python3 src/make_social.py         # -> website/assets/og-image.png

echo "[4/6] Building the paper .................................."
( cd paper && tectonic aegis.tex ) # -> paper/aegis.pdf
cp paper/aegis.pdf paper/aegis.tex website/assets/

echo "[5/6] Running the test suite ............................."
python3 -m pytest -q

echo "[6/6] Done. Open website/index.html (or visit the live site)."
