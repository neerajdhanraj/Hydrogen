#!/usr/bin/env python3
"""Regenerate the manuscript figures from released result tables."""
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
for script in [
    "make_figures_1_to_4.py",
    "make_figure_5.py",
    "make_figure_6.py",
    "make_supplementary_figures.py",
]:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)
