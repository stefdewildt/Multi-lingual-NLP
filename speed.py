"""How much faster fastdetect is than detectgpt, from the time score.py
recorded for every text (score_seconds in scores.csv):

    python speed.py
    python speed.py --scores outputs/scores

Make sure that results we're found on the same device."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluation.data import group_runs, merged_scores, read_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=Path("outputs/scores"), help="score.py's output folder")
    args = parser.parse_args()

    rows = []
    for detector, runs in group_runs(read_runs(args.scores)).items():
        if detector.role != "main" or detector.method == "baseline":
            continue
        hosts = {entry["created_by"].split("@")[-1] for run in runs for entry in run.meta["runs"]}
        if not all("snellius" in host for host in hosts):
            continue
        scores = merged_scores(detector, runs)
        for dataset, part in scores.groupby("dataset", observed=True):
            rows.append({"model": (detector.config.scoring or "").split("/")[-1],
                         "dataset": str(dataset), "method": detector.method,
                         "seconds": float(part["score_seconds"].median())})
    table = pd.DataFrame(rows).pivot_table(index=["model", "dataset"], columns="method", values="seconds")
    table = table.dropna(subset=["detectgpt", "fastdetect"])
    table["detectgpt / fastdetect"] = table["detectgpt"] / table["fastdetect"]
    with pd.option_context("display.width", 120):
        print("Median seconds per text, and how many times longer detectgpt took:\n")
        print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nfastdetect is {table['detectgpt / fastdetect'].min():.1f}-{table['detectgpt / fastdetect'].max():.1f}x "
          f"faster (median {table['detectgpt / fastdetect'].median():.1f}x).")


if __name__ == "__main__":
    main()
