"""Run Fast-DetectGPT on the MULTITuDE dataset"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

# make local imports work when this file is run directly
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
DATASETS_DIR = REPO_ROOT / "datasets"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(DATASETS_DIR))

import tqdm
from metrics import get_precision_recall_metrics, get_roc_metrics
from model import load_model, load_tokenizer
from multitude import filter_dataframe, load_dataframe

from detector.fast import FastDetector

DEFAULT_CSV = REPO_ROOT / "datasets" / "MULTITuDE" / "multitude.csv"


def read_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load the selected MULTITuDE rows."""

    dataframe = load_dataframe(args.csv_path)

    # apply the filters for language and generator models
    dataframe = filter_dataframe(
        dataframe,
        split=args.split,
        languages=args.languages or None,
        models=args.generator_models or None,
    )
    rows = dataframe.to_dict(orient="records")

    human = [row for row in rows if int(row["label"]) == 0]
    machine = [row for row in rows if int(row["label"]) == 1]

    # balances the labels for small test runs if they are unbalanced
    if args.max_samples_per_class > 0:
        human = human[:args.max_samples_per_class]
        machine = machine[:args.max_samples_per_class]

    return human + machine


def make_detector(args: argparse.Namespace) -> FastDetector:
    """Load the models and build the detector."""

    # load the scoring model and its tokenizer
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()

    if args.sampling_model_name == args.scoring_model_name:
        # reuse the model if the sampling and reference model are the same
        reference_tokenizer = scoring_tokenizer
        reference_model = scoring_model
    else:
        reference_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
        reference_model = load_model(args.sampling_model_name, args.device, args.cache_dir)
        reference_model.eval()

    return FastDetector(
        reference_model=reference_model,
        reference_tokenizer=reference_tokenizer,
        scoring_model=scoring_model,
        scoring_tokenizer=scoring_tokenizer,
        mode="analytic",
        n_samples=None,
        sample_top_p=None,
        sample_top_k=None,
        device=args.device,
    )


def write_results_csv(output_file: Path, predictions: list[dict[str, object]], roc_auc: float, pr_auc: float) -> None:
    """Write results into a CSV file."""

    # the field names used for the csv
    fieldnames = [
        "sample_index",
        "score",
        "label",
        "language",
        "multi_label",
        "roc_auc",
        "pr_auc",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for sample_index, prediction in enumerate(predictions):
            writer.writerow(
                {
                    "sample_index": sample_index,
                    **prediction,
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                }
            )


def main(args: argparse.Namespace) -> None:
    # select the rows for this run based on the filters
    rows = read_rows(args)
    if not rows:
        raise ValueError("No MULTITuDE rows matched the requested filters.")

    labels: list[int] = []
    scores: list[float] = []
    predictions: list[dict[str, object]] = []
    detector = make_detector(args)

    for row in tqdm.tqdm(rows, desc="Scoring MULTITuDE"):
        score = detector.score(row["text"])
        label = int(row["label"])
        labels.append(label)
        scores.append(score)
        predictions.append(
            {
                "score": score,
                "label": label,
                "language": row["language"],
                "multi_label": row["multi_label"],
            }
        )

    # split scores by label for the metrics
    human_scores = [score for score, label in zip(scores, labels) if label == 0]
    machine_scores = [score for score, label in zip(scores, labels) if label == 1]
    if not human_scores or not machine_scores:
        raise ValueError("Both human and machine-written rows are required for AUROC.")

    _, _, roc_auc = get_roc_metrics(human_scores, machine_scores)
    _, _, pr_auc = get_precision_recall_metrics(human_scores, machine_scores)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    write_results_csv(args.output_file, predictions, roc_auc, pr_auc)

    print(f"Samples: {len(rows)}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC:  {pr_auc:.4f}")
    print(f"Results written to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV, help="Path to multitude.csv")
    parser.add_argument("--output_file", type=Path, required=True, help="Where to write the CSV results")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate")
    parser.add_argument("--languages", nargs="*", help="Optional ISO language codes")
    parser.add_argument("--generator_models", nargs="*", help="Optional MULTITuDE generator names")
    parser.add_argument("--max_samples_per_class", type=int, default=0, help="Maximum human and machine samples per language; 0 means all")
    parser.add_argument("--sampling_model_name", required=True, help="Reference model name")
    parser.add_argument("--scoring_model_name", required=True, help="Scoring model name")
    parser.add_argument("--device", default="cuda", help="Torch device: cuda or cpu")
    parser.add_argument("--cache_dir", default=str(REPO_ROOT / "cache"), help="Hugging Face model cache")
    main(parser.parse_args())
