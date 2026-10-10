"""Draws the report figures from the scores score.py wrote:

    python eval.py
    python eval.py --scores outputs/scores --output outputs/evaluation/figures
    python eval.py --figures  # also update the figures in ./figures

See evaluation/figures.py for the in-text figures, evaluation/appendix.py
for the appendix figures, evaluation/data.py for which texts every detector
is evaluated on, and evaluation/auc.py for the adjusted AUC.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from evaluation import appendix
from evaluation.data import load
from evaluation import figures
from evaluation.style import apply_style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=Path("outputs/scores"), help="score.py's output folder")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/figures"), help="where the figures go")
    parser.add_argument("--figures", action="store_true", help="also write the figures to ./figures")
    args = parser.parse_args()
    apply_style()
    view = load(args.scores)
    figures.draw(view, args.output)
    appendix.draw(view, args.output / "appendix")
    if args.figures:
        for png in args.output.rglob("*.png"):
            target = Path("figures") / png.relative_to(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(png, target)
    print(f"figures: {args.output}/")


if __name__ == "__main__":
    main()
