"""

Scores every document in a dataset with one detector, writing the
generatedness (d-)score per row. Inference only, no metrics.

    python score.py --detector fastdetect --dataset multitude
        --scoring gpt2 --fastdetect-reference gpt2 --fastdetect-mode analytic
    python score.py --detector detectgpt --dataset multisocial
        --scoring gpt2 --detectgpt-mask t5-small --detectgpt-n-perturbations 10
    python score.py --detector baseline --dataset multitude
        --baseline-weights detector/baseline/models/multitude_all

Each run creates a folder in --output, named after it's detector config.
Share these over WeTransfer/Drive and don't commit them.

    outputs/scores/<config>/scores.csv
    outputs/scores/<config>/meta.json

scores.csv has one row per (dataset, row_id): text, the dataset's own 
metadata columns, score, error, score_seconds, etc. 

meta.json records the detector config plus one entry per `python score.py` 
invocation into that folder. 

Running the same detector config but for different datasets, seeds, etc
will add new entries to scores.csv and meta.json.

Re-running the exact same command resumes previously left of work,
so adding the missing entries to scores.csv, and adds a new entry to 
meta.json. See --overwrite and --retry-errors (python score.py --help) 
for details.

"""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import socket
import subprocess
import sys
import time
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from datasets import multisocial, multitude
from detector.baseline.baseline_detector import BaselineDetector
from detector.fast import FastDetector, Mode as FastMode
from detector.models import load_model, load_tokenizer
from detector.perturbation import (
    CurvatureNormalization,
    Metric,
    PerturbationDetector,
)

REPO_ROOT = Path(__file__).resolve().parent
DATASET_MODULES = {"multitude": multitude, "multisocial": multisocial}
BASE_COLUMNS = ["text", "label", "multi_label", "split", "language", "length", "source"]

Detector = Literal["fastdetect", "detectgpt", "baseline"]
Dataset = Literal["multitude", "multisocial"]
DatasetSplit = Literal["all", "train", "test"]
Device = Literal["auto", "cpu", "cuda", "mps"]
RowValue = str | int | float | bool | None  # everything a scores.csv cell can hold


@dataclass(frozen=True)
class ScoreArgs:
    """Arguments to this cli script"""

    detector: Detector  # which detector implementation to run
    dataset: Dataset  # which dataset to score
    output: Path  # folder under which the run's config-named subfolder is written
    tag: str | None  # free-text label appended to the run's output folder name

    dataset_split: DatasetSplit  # "all", "train", or "test"
    dataset_languages: list[str] | None  # restrict to these ISO codes, e.g. ["nl", "en"]
    dataset_models: list[str] | None  # restrict to these multi_label values
    multisocial_exclude_noise: bool  # MultiSocial only: drop potential_noise=1 rows
    dataset_csv_path: Path | None  # override the dataset loader's default csv path
    dataset_limit: int | None  # (testing) only score N matching rows, shuffled first so this isn't single-class/single-language
    dataset_limit_language: int | None  # (testing) cap each language at N rows, keep all rows for languages with fewer
    dataset_sample: int | None  # (testing) randomly subsample N matching rows

    baseline_weights: Path | None  # baseline: save_dir of a trained baseline

    fastdetect_reference: str | None  # fastdetect: reference model id, defaults to scoring
    fastdetect_mode: FastMode  # fastdetect: "analytic" (closed-form) or "sampling" (Monte Carlo)
    fastdetect_n_samples: int | None  # fastdetect: draws for sampling mode

    detectgpt_mask: str | None  # detectgpt: seq2seq mask-filling model id, e.g. t5-small
    detectgpt_n_perturbations: int  # detectgpt: perturbed neighbors averaged per score
    detectgpt_metric: Metric  # detectgpt: "sum" or "average" token log-likelihood
    detectgpt_normalize_by: CurvatureNormalization  # detectgpt: "std" or "none"
    detectgpt_span_length: int | None  # detectgpt: tokens per masked span
    detectgpt_pct_masked: float | None  # detectgpt: fraction of the text masked

    scoring: str | None  # fastdetect/detectgpt: huggingface model id to score text under
    top_p: float | None  # fastdetect (either mode) / detectgpt mask filling: nucleus cutoff
    top_k: int | None  # fastdetect (either mode) / detectgpt mask filling: top-k cutoff

    device: Device  # "auto", "cpu", "cuda", or "mps"
    hf_cache_dir: Path | None  # huggingface cache dir
    seed: int  # random seed for sampling/subsampling and detector randomness
    save_every: int  # rows between disk flushes
    overwrite: bool  # discard any existing output for this config before starting
    retry_errors: bool  # on resume, also re-attempt rows that errored last time
    dry_run: bool  # print what would run, load nothing, score nothing


class RunEntry(TypedDict):
    """One `python score.py` invocation into a config folder."""

    dataset: str
    dataset_split: str
    dataset_languages: list[str] | None
    dataset_models: list[str] | None
    dataset_csv_path: str
    dataset_csv_sha256: str | None
    dataset_sample: int | None
    dataset_limit: int | None
    dataset_limit_language: int | None
    complete: bool
    device: str
    hardware: str
    created_by: str
    git_commit: str | None
    package_versions: dict[str, str]
    started_at: str
    finished_at: str
    duration_seconds: float
    rows_matching_filters: int
    rows_already_done_before_run: int
    rows_scored_this_run: int
    rows_errored_this_run: int


class ScoresMeta(TypedDict):
    """The meta.json file saved in a `outputs/scores/<detector> folder. 
    eval.py uses this too. detector_config is constant for the folder, 
    runs has one entry per `python score.py` invocation into it."""

    detector_config: DetectorConfigDict
    runs: list[RunEntry]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Progress so far was already flushed to disk, "
            "re-run the same command to resume."
        )
        sys.exit(130)


def run(args: ScoreArgs) -> None:
    """Finds all generatedness scores / d-scores for every document in
    the selected document using the selected detection config."""

    dataset_csv_path = _resolve_dataset_csv_path(args)

    # dataset loading
    dataset_df = _load_data(args, dataset_csv_path)
    # fixed regardless of args.dataset: a folder can hold rows from more than
    # one dataset, so every dataset must write the same columns to scores.csv
    output_columns = BASE_COLUMNS + ["potential_noise"]

    # output folder setup
    run_folder = _build_run_folder_name(args)
    run_dir = args.output / run_folder
    run_dir.mkdir(parents=True, exist_ok=True)
    output_scores_csv_path = run_dir / "scores.csv"
    output_meta_json_path = run_dir / "meta.json"
    if args.overwrite:
        output_scores_csv_path.unlink(missing_ok=True)
        output_meta_json_path.unlink(missing_ok=True)

    # for resuming unfinished runs, keyed by (dataset, row_id): a folder can
    # hold rows from more than one --dataset
    already_done: set[tuple[str, int]] = set()
    if output_scores_csv_path.exists():
        existing = pd.read_csv(
            output_scores_csv_path, usecols=["dataset", "row_id", "error"]
        )
        if args.retry_errors:
            existing = existing.loc[existing["error"].fillna("") == ""]
        already_done = {
            (str(ds), int(row_id))
            for ds, row_id in zip(existing["dataset"], existing["row_id"])
        }
    todo_mask = [
        (args.dataset, int(row_id)) not in already_done for row_id in dataset_df.index
    ]
    todo = dataset_df.loc[todo_mask]
    print(
        f"{args.dataset}/{args.detector}: {len(dataset_df)} rows match filters, "
        f"{len(dataset_df) - len(todo)} already scored, {len(todo)} left to do"
    )
    print(f"  -> {output_scores_csv_path}")
    if len(todo) == 0:
        print("Nothing to do.")
        return

    # dry run mode doesn't do any scoring
    if args.dry_run:
        print("(--dry-run: stopping before loading any models)")
        return

    # build the detector that's used for scoring
    device = _default_device() if args.device == "auto" else args.device
    print(f"Loading models on device={device} ...")
    scorer = _build_scorer(args, device)

    # other setup things
    fieldnames = ["dataset", "row_id"] + output_columns + ["score", "error", "score_seconds"]
    write_header = not output_scores_csv_path.exists()
    if not write_header:
        with output_scores_csv_path.open(encoding="utf-8") as existing:
            assert existing.readline().strip() == ",".join(fieldnames), f"scores.csv header mismatch: {output_scores_csv_path}"
    started_at = datetime.now(timezone.utc)
    n_errors = 0

    with output_scores_csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, (row_id, row) in enumerate(
            tqdm(todo.iterrows(), total=len(todo), desc=run_folder)
        ):
            text = row["text"]
            record: dict[str, RowValue] = {
                "dataset": args.dataset,
                "row_id": cast(int, row_id),
                **{
                    col: cast(RowValue, row[col]) if col in dataset_df.columns else None
                    for col in output_columns
                },
            }
            start = time.perf_counter()
            try:
                _seed_row(args.seed + cast(int, row_id), device)
                record["score"] = scorer.score(text)
                record["error"] = ""
            except Exception as exc:
                record["score"] = ""
                record["error"] = f"{type(exc).__name__}: {exc}"
                n_errors += 1
            record["score_seconds"] = time.perf_counter() - start
            writer.writerow(record)

            if (i + 1) % args.save_every == 0:
                file.flush()

    finished_at = datetime.now(timezone.utc)

    # append this invocation's entry to the folder's meta.json
    previous_runs: list[RunEntry] = []
    if output_meta_json_path.exists():
        previous_runs = cast(ScoresMeta, json.loads(output_meta_json_path.read_text()))["runs"]
    new_entry: RunEntry = {
        "dataset": args.dataset,
        "dataset_split": args.dataset_split,
        "dataset_languages": args.dataset_languages,
        "dataset_models": args.dataset_models,
        "dataset_csv_path": str(dataset_csv_path),
        "dataset_csv_sha256": (
            _file_checksum(dataset_csv_path) if dataset_csv_path.exists() else None
        ),
        "dataset_sample": args.dataset_sample,
        "dataset_limit": args.dataset_limit,
        "dataset_limit_language": args.dataset_limit_language,
        # False when --dataset-sample/--dataset-limit/--dataset-limit-language were used:
        # this run doesn't cover the full filtered dataset, so eval.py warns before
        # including it.
        "complete": (
            args.dataset_sample is None
            and args.dataset_limit is None
            and args.dataset_limit_language is None
        ),
        "device": device,
        "hardware": _hardware_info(device),
        "created_by": f"{getpass.getuser()}@{socket.gethostname()}",
        "git_commit": _git_commit(),
        "package_versions": _package_versions(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "rows_matching_filters": len(dataset_df),
        "rows_already_done_before_run": len(dataset_df) - len(todo),
        "rows_scored_this_run": len(todo),
        "rows_errored_this_run": n_errors,
    }
    prior_checksums = {
        r["dataset_csv_sha256"] for r in previous_runs
        if r["dataset"] == args.dataset and r["dataset_csv_sha256"]
    }
    if prior_checksums and new_entry["dataset_csv_sha256"] not in prior_checksums:
        print(f"WARNING: {args.dataset} csv checksum differs from a previous run in {run_dir}")
    metadata: ScoresMeta = {
        "detector_config": cast(DetectorConfigDict, asdict(_detector_config(args))),
        "runs": [*previous_runs, new_entry],
    }
    output_meta_json_path.write_text(json.dumps(metadata, indent=2))

    print(
        f"Done: {len(todo)} rows scored ({n_errors} errors) in {new_entry['duration_seconds']:.1f}s"
    )
    print(f"  csv:  {output_scores_csv_path}")
    print(f"  meta: {output_meta_json_path}")


def parse_args(argv: list[str] | None = None) -> ScoreArgs:
    parser = argparse.ArgumentParser(
        description="Score every document in a dataset with one detector config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--detector", required=True, choices=["fastdetect", "detectgpt", "baseline"]
    )
    parser.add_argument(
        "--dataset", required=True, choices=["multitude", "multisocial"]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/scores"),
        help="folder to write results into",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="free-text label appended to the run's output folder name, e.g. your name",
    )

    parser.add_argument(
        "--dataset-split", default="all", choices=["all", "train", "test"]
    )
    parser.add_argument(
        "--dataset-languages",
        nargs="+",
        default=None,
        help="e.g. --dataset-languages nl en",
    )
    parser.add_argument(
        "--dataset-models",
        nargs="+",
        default=None,
        help="filter multi_label, e.g. --dataset-models human gpt-4",
    )
    parser.add_argument(
        "--multisocial-exclude-noise",
        action="store_true",
        help="MultiSocial only: drop rows the dataset itself flags as potentially "
        "noisy/low-quality (potential_noise=1), see datasets/MultiSocial/README.md",
    )
    parser.add_argument(
        "--dataset-csv-path",
        type=Path,
        default=None,
        help="override the dataset loader's default csv path",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="(testing) only score N matching rows, shuffled first, marks the run's meta.json incomplete",
    )
    parser.add_argument(
        "--dataset-limit-language",
        type=int,
        default=None,
        help="(testing) cap each language at N rows (keep all rows for languages with fewer), "
        "marks the run's meta.json incomplete",
    )
    parser.add_argument(
        "--dataset-sample",
        type=int,
        default=None,
        help="(testing) randomly subsample N matching rows before scoring, marks the run's meta.json incomplete",
    )

    parser.add_argument(
        "--baseline-weights",
        type=Path,
        default=None,
        help="baseline: save_dir of a trained baseline",
    )

    parser.add_argument(
        "--fastdetect-reference",
        default=None,
        help="fastdetect: reference huggingface model id, defaults to --scoring",
    )
    parser.add_argument(
        "--fastdetect-mode",
        default="analytic",
        choices=["analytic", "sampling"],
        help="fastdetect: analytic is the exact closed-form estimate, "
        "sampling is a Monte Carlo estimate using --fastdetect-n-samples draws",
    )
    parser.add_argument(
        "--fastdetect-n-samples",
        type=int,
        default=None,
        help="fastdetect: number of Monte Carlo draws, required when --fastdetect-mode sampling",
    )

    parser.add_argument(
        "--detectgpt-mask",
        default=None,
        help="detectgpt: seq2seq mask-filling model id, e.g. t5-small",
    )
    parser.add_argument(
        "--detectgpt-n-perturbations",
        type=int,
        default=10,
        help="detectgpt: how many perturbed neighbor texts to average the metric over",
    )
    parser.add_argument(
        "--detectgpt-metric",
        default="average",
        choices=["sum", "average"],
        help="detectgpt: sum is the total log-likelihood, average is per-token "
        "(fixes a sequence-length bias, and matches what fastdetect's own reference code computes)",
    )
    parser.add_argument(
        "--detectgpt-normalize-by",
        default="std",
        choices=["std", "none"],
        help="detectgpt: std divides the curvature by the perturbed scores' std (Mitchell et al.), "
        "none doesn't (Mireshghallah et al., Bao et al.)",
    )
    parser.add_argument(
        "--detectgpt-span-length",
        type=int,
        default=2,  # PerturbationDetector's own default
        help="detectgpt: tokens per masked span",
    )
    parser.add_argument(
        "--detectgpt-pct-masked",
        type=float,
        default=0.15,  # PerturbationDetector's own default
        help="detectgpt: fraction of the text to mask, spread over spans of --detectgpt-span-length",
    )

    parser.add_argument(
        "--scoring",
        "--fastdetect-scoring",
        "--detectgpt-scoring",
        dest="scoring",
        default=None,
        help="fastdetect/detectgpt: huggingface model id to score text under (required)",
    )
    parser.add_argument(
        "--top-p",
        "--fastdetect-top-p",
        "--detectgpt-top-p",
        dest="top_p",
        type=float,
        default=1.0,  # no filtering, for both fastdetect and PerturbationDetector's own default
        help="fastdetect (either mode) / detectgpt mask filling: nucleus sampling cutoff",
    )
    parser.add_argument(
        "--top-k",
        "--fastdetect-top-k",
        "--detectgpt-top-k",
        dest="top_k",
        type=int,
        default=None,
        help="fastdetect (either mode) / detectgpt mask filling: top-k cutoff",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="auto picks cuda/mps/cpu, in that order, based on what's available",
    )
    parser.add_argument(
        "--hf-cache-dir", type=Path, default=None, help="huggingface cache dir"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-every", type=int, default=50, help="rows between disk flushes"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="discard any existing output for this config",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="on resume, also re-attempt rows that errored last time (default: leave them as-is)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would run, load nothing, score nothing",
    )

    parsed = parser.parse_args(argv)
    args = ScoreArgs(**vars(parsed))

    if args.detector in ("fastdetect", "detectgpt") and not args.scoring:
        parser.error(f"--scoring is required for --detector {args.detector}")
    if args.detector == "detectgpt" and not args.detectgpt_mask:
        parser.error("--detectgpt-mask is required for --detector detectgpt")
    if args.detector == "baseline" and not args.baseline_weights:
        parser.error("--baseline-weights is required for --detector baseline")
    if (
        args.detector == "fastdetect"
        and args.fastdetect_mode == "sampling"
        and args.fastdetect_n_samples is None
    ):
        parser.error(
            "--fastdetect-n-samples is required when --fastdetect-mode sampling"
        )
    if args.multisocial_exclude_noise and args.dataset != "multisocial":
        parser.error(
            "--multisocial-exclude-noise only applies to --dataset multisocial"
        )
    if sum(x is not None for x in (args.dataset_limit, args.dataset_sample, args.dataset_limit_language)) > 1:
        parser.error("--dataset-limit, --dataset-sample and --dataset-limit-language are mutually exclusive")

    return args


def _build_scorer(
    args: ScoreArgs, device: str
) -> BaselineScoreDetector | FastDetector | PerturbationDetector:
    if args.detector == "baseline":
        assert args.baseline_weights is not None
        return BaselineScoreDetector(BaselineDetector.load(args.baseline_weights))

    if args.detector == "fastdetect":
        assert args.scoring is not None
        scoring_model = load_model(args.scoring, "causal", device, args.hf_cache_dir)
        scoring_tokenizer = load_tokenizer(args.scoring, args.hf_cache_dir)
        if args.fastdetect_reference and args.fastdetect_reference != args.scoring:
            reference_model = load_model(
                args.fastdetect_reference, "causal", device, args.hf_cache_dir
            )
            reference_tokenizer = load_tokenizer(
                args.fastdetect_reference, args.hf_cache_dir
            )
        else:
            reference_model, reference_tokenizer = scoring_model, scoring_tokenizer
        return FastDetector(
            reference_model=reference_model,
            reference_tokenizer=reference_tokenizer,
            scoring_model=scoring_model,
            scoring_tokenizer=scoring_tokenizer,
            mode=args.fastdetect_mode,
            n_samples=args.fastdetect_n_samples,
            top_p=args.top_p,
            top_k=args.top_k,
            device=device,
        )

    if args.detector == "detectgpt":
        assert args.scoring is not None
        assert args.detectgpt_mask is not None
        assert args.detectgpt_span_length is not None
        assert args.detectgpt_pct_masked is not None
        assert args.top_p is not None
        scoring_model = load_model(args.scoring, "causal", device, args.hf_cache_dir)
        scoring_tokenizer = load_tokenizer(args.scoring, args.hf_cache_dir)
        mask_model = load_model(
            args.detectgpt_mask, "seq2seq", device, args.hf_cache_dir
        )
        mask_tokenizer = load_tokenizer(args.detectgpt_mask, args.hf_cache_dir)
        return PerturbationDetector(
            scoring_model=scoring_model,
            scoring_tokenizer=scoring_tokenizer,
            mask_model=mask_model,
            mask_tokenizer=mask_tokenizer,
            device=device,
            n_perturbations=args.detectgpt_n_perturbations,
            metric=args.detectgpt_metric,
            normalize_by=args.detectgpt_normalize_by,
            span_length=args.detectgpt_span_length,
            pct_masked=args.detectgpt_pct_masked,
            top_p=args.top_p,
            top_k=args.top_k,
        )

    raise ValueError(f"unknown detector {args.detector!r}")


def _seed_row(seed: int, device: str) -> None:
    """Reseeds every RNG a detector can draw from before scoring one row, so
    a score depends only on (detector config, seed, row_id, text), not on
    which other rows were scored before it in the same run. torch.manual_seed
    covers torch's CPU/CUDA generators (fastdetect sampling mode) but not
    MPS's separate generator or numpy's global RNG (detectgpt's masking)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)


def _load_data(args: ScoreArgs, csv_source_path: Path) -> pd.DataFrame:
    """Returns the pandas dataframe containing all rows that need to be scored."""
    module = DATASET_MODULES[args.dataset]
    frame = module.load_dataframe(csv_source_path)
    filter_kwargs: dict[str, Any] = {
        "split": None if args.dataset_split == "all" else args.dataset_split,
        "languages": args.dataset_languages,
        "models": args.dataset_models,
    }
    if args.dataset == "multisocial":
        filter_kwargs["exclude_noise"] = args.multisocial_exclude_noise
    frame = module.filter_dataframe(frame, **filter_kwargs)

    if args.dataset_limit_language is not None:
        frame = pd.concat(
            group if len(group) <= args.dataset_limit_language
            else group.sample(n=args.dataset_limit_language, random_state=args.seed)
            for _, group in frame.groupby("language")
        )
    if args.dataset_sample is not None:
        frame = frame.sample(
            n=min(args.dataset_sample, len(frame)), random_state=args.seed
        )
    if args.dataset_limit is not None:
        frame = frame.sample(frac=1, random_state=args.seed).iloc[: args.dataset_limit]
    return frame.sort_index()

def _build_run_folder_name(args: ScoreArgs) -> str:
    """Creates the name for the folder where scores.csv and meta.json
    are outputted. This name is unique for every unique detector config.
    The detector config has the property that all thescores produced by
    some config cannot be different from the scores produced by another
    run with the same config"""
    slug = str(_detector_config(args))
    if args.tag:
        slug += f"_{_sanitize(args.tag)}"
    return slug


@dataclass
class BaselineScoreDetector:
    """So baseline detector also has the score method"""

    detector: BaselineDetector

    def score(self, text: str) -> float:
        features = self.detector.vectorizer.transform([text])
        return float(self.detector.model.predict_proba(features)[0, 1])


def _sanitize(value: str) -> str:
    return "".join(c if (c.isalnum() or c in "-._") else "-" for c in value)


@dataclass(frozen=True)
class DetectorConfig:
    """Two configs that compare equal would score identically, which is what
    eval.py uses to decide which run folders to merge as one detector.
    Frozen, so equality and hashing (for use as a dict key) come for free.

    Deliberately excludes dataset: the same text must score the same
    regardless of which dataset it came from, so a config folder can hold
    rows from multiple datasets (scores.csv has a dataset column instead).

    Includes seed, since the stochastic detectors (fastdetect sampling,
    detectgpt masking) draw from it. Each row is scored with its own
    seed + row_id (see _seed_row), so the seed here plus a row_id is enough
    to make "same config + same text -> same score" hold exactly.

    Only the fields relevant to args.detector are set. The rest stay None.
    """

    detector: Detector
    seed: int

    # fastdetect
    scoring: str | None = None
    fastdetect_reference: str | None = None  # never None for fastdetect: falls back to scoring
    fastdetect_mode: FastMode | None = None
    fastdetect_n_samples: int | None = None  # only set in sampling mode

    # detectgpt
    detectgpt_mask: str | None = None
    detectgpt_metric: Metric | None = None
    detectgpt_normalize_by: CurvatureNormalization | None = None
    detectgpt_n_perturbations: int | None = None
    detectgpt_span_length: int | None = None
    detectgpt_pct_masked: float | None = None

    # fastdetect (either mode) / detectgpt mask filling
    top_p: float | None = None
    top_k: int | None = None

    # baseline
    baseline_weights_path: str | None = None  # resolved absolute path
    baseline_weights_hash: str | None = None  # sha256 of vectorizer.pkl + model.pkl

    def __str__(self) -> str:
        parts: list[str] = [self.detector]
        if self.detector == "fastdetect":
            assert self.scoring is not None
            assert self.fastdetect_reference is not None
            assert self.fastdetect_mode is not None
            parts.append(f"score-{_sanitize(self.scoring)}")
            parts.append(f"ref-{_sanitize(self.fastdetect_reference)}")
            parts.append(self.fastdetect_mode)
            if self.fastdetect_mode == "sampling":
                parts.append(f"n{self.fastdetect_n_samples}")
            if self.top_p is not None:
                parts.append(f"topp{self.top_p}")
            if self.top_k is not None:
                parts.append(f"topk{self.top_k}")
        if self.detector == "detectgpt":
            assert self.scoring is not None
            assert self.detectgpt_mask is not None
            parts.append(f"score-{_sanitize(self.scoring)}")
            parts.append(f"mask-{_sanitize(self.detectgpt_mask)}")
            parts.append(f"{self.detectgpt_metric}-{self.detectgpt_normalize_by}")
            parts.append(f"p{self.detectgpt_n_perturbations}")
            parts.append(f"span{self.detectgpt_span_length}")
            parts.append(f"pm{self.detectgpt_pct_masked}")
            parts.append(f"topp{self.top_p}")
            if self.top_k is not None:
                parts.append(f"topk{self.top_k}")
        if self.detector == "baseline":
            assert self.baseline_weights_path is not None
            parts.append(f"weights-{_sanitize(Path(self.baseline_weights_path).name)}")
            if self.baseline_weights_hash:
                parts.append(self.baseline_weights_hash[:8])
        parts.append(f"seed{self.seed}")
        return "_".join(parts)


class DetectorConfigDict(TypedDict):
    """DetectorConfig, as stored in meta.json"""

    detector: Detector
    seed: int
    scoring: str | None
    fastdetect_reference: str | None
    fastdetect_mode: FastMode | None
    fastdetect_n_samples: int | None
    detectgpt_mask: str | None
    detectgpt_metric: Metric | None
    detectgpt_normalize_by: CurvatureNormalization | None
    detectgpt_n_perturbations: int | None
    detectgpt_span_length: int | None
    detectgpt_pct_masked: float | None
    top_p: float | None
    top_k: int | None
    baseline_weights_path: str | None
    baseline_weights_hash: str | None


def _relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _detector_config(args: ScoreArgs) -> DetectorConfig:
    """Creates the detector config from cli arguments.
    The detector config has the property that the same
    config must always give the same scores for the 
    same text. So therefore we use it to e.g. design
    unique folder names and make sure to aggregate 
    score data created with the same config.
    """
    if args.detector == "fastdetect":
        assert args.scoring is not None
        sampling = args.fastdetect_mode == "sampling"
        return DetectorConfig(
            detector=args.detector,
            seed=args.seed,
            scoring=args.scoring,
            fastdetect_reference=args.fastdetect_reference or args.scoring,
            fastdetect_mode=args.fastdetect_mode,
            fastdetect_n_samples=args.fastdetect_n_samples if sampling else None,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    if args.detector == "detectgpt":
        assert args.scoring is not None
        assert args.detectgpt_mask is not None
        assert args.detectgpt_span_length is not None
        assert args.detectgpt_pct_masked is not None
        assert args.top_p is not None
        return DetectorConfig(
            detector=args.detector,
            seed=args.seed,
            scoring=args.scoring,
            detectgpt_mask=args.detectgpt_mask,
            detectgpt_metric=args.detectgpt_metric,
            detectgpt_normalize_by=args.detectgpt_normalize_by,
            detectgpt_n_perturbations=args.detectgpt_n_perturbations,
            detectgpt_span_length=args.detectgpt_span_length,
            detectgpt_pct_masked=args.detectgpt_pct_masked,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    if args.detector == "baseline":
        assert args.baseline_weights is not None
        return DetectorConfig(
            detector=args.detector,
            seed=args.seed,
            baseline_weights_path=_relative_or_absolute(args.baseline_weights),
            baseline_weights_hash=_baseline_weights_checksum(args.baseline_weights),
        )
    raise ValueError(f"unknown detector {args.detector!r}")


def _resolve_dataset_csv_path(args: ScoreArgs) -> Path:
    return (
        args.dataset_csv_path
        if args.dataset_csv_path
        else DATASET_MODULES[args.dataset].DEFAULT_CSV_PATH
    )


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _hardware_info(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return platform.processor() or platform.machine() or "unknown"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        return None


def _package_versions() -> dict[str, str]:
    versions = {"torch": str(torch.__version__)}
    for name in ("transformers", "sklearn"):
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            pass
    return versions


def _baseline_weights_checksum(save_dir: Path) -> str | None:
    """sha256 of the trained weight files in a baseline save_dir, so two
    people pointing --baseline-weights at directories with the same name
    don't get silently merged if the weights inside differ."""
    files = [save_dir / "vectorizer.pkl", save_dir / "model.pkl"]
    if not all(file.exists() for file in files):
        return None
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.read_bytes())
    return digest.hexdigest()


def _file_checksum(path: Path) -> str:
    """sha256 of the raw dataset csv actually used for this run, so two
    people's outputs can be checked for having scored the exact same file
    (not just a file with the same name)."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
