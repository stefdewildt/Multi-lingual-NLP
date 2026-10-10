"""Reading the scores that score.py wrote (outputs/scores/**/<config>/, nested
folders too) into one View: per detector and dataset the texts it is
evaluated on, with their scores and lengths, and the subgroup AUCs the
adjusted AUC is computed from (see auc.py)."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from evaluation.auc import subgroup_aucs
from evaluation.groups import ScoringModel, baseline_name, baseline_trained_on, role, scoring_model
from score import DATASET_MODULES, DetectorConfig, ScoresMeta
from utils.checksum import file_checksum
from utils.length import LENGTH_TOKENIZER_MODEL, cached_token_lengths

SCORE_COLUMNS = ["dataset", "row_id", "label", "multi_label", "split", "language", "score", "error", "score_seconds"]
"""The scores.csv columns we need. The texts themselves are read once from the dataset csvs instead."""

CATEGORY_COLUMNS = ["dataset", "multi_label", "split", "language"]

MIN_COVERAGE = 0.99
"""A detector is evaluated on a dataset if it scored at least this share of the texts it should have."""


@dataclass(frozen=True)
class Detector:
    """One detector config, under a short readable label. Folders with the
    same config (e.g. a run continued elsewhere) are the same detector."""

    config: DetectorConfig
    label: str

    @property
    def key(self) -> str:
        return str(self.config)

    @property
    def method(self) -> str:
        """fastdetect, detectgpt or baseline."""
        return self.config.detector

    @property
    def role(self) -> str:
        return role(self.config)

    @property
    def model(self) -> ScoringModel:
        """The scoring model of a zero-shot detector (fastdetect, detectgpt)."""
        model = scoring_model(self.config.scoring)
        if model is None:
            raise ValueError(f"{self.label} has no scoring model in evaluation/groups.py")
        return model


@dataclass
class Run:
    """One outputs/scores/**/<config>/ folder: its meta.json and scores.csv."""

    dir: Path
    meta: ScoresMeta
    config: DetectorConfig
    scores: pd.DataFrame  # one row per (dataset, row_id), rows that errored dropped
    errors: pd.DataFrame  # the rows that errored
    truncated: bool  # scores.csv ended mid-row, e.g. copied while still being written


@dataclass
class Dataset:
    frame: pd.DataFrame  # the dataset csv's text and split columns, indexed by row_id
    checksum: str


@dataclass
class View:
    """The texts every detector is evaluated on, and their subgroup AUCs."""

    frames: dict[Detector, dict[str, pd.DataFrame]]  # per detector and dataset: its rows, with a computed_length
    cells: pd.DataFrame  # subgroup_aucs of every detector and dataset, with a detector (key) and dataset column
    coverage: pd.DataFrame  # per detector and dataset: rows scored, rows expected, whether it is included

    @property
    def detectors(self) -> list[Detector]:
        return [detector for detector, frames in self.frames.items() if frames]

    @property
    def datasets(self) -> list[str]:
        return sorted({dataset for frames in self.frames.values() for dataset in frames})


def load(scores_dir: Path) -> View:
    """The view of every main detector, see build_view."""
    runs = read_runs(scores_dir)
    detectors = {d: runs_ for d, runs_ in group_runs(runs).items() if d.role == "main"}
    scored = {detector: merged_scores(detector, runs_) for detector, runs_ in detectors.items()}
    datasets = load_datasets({str(d) for frame in scored.values() for d in frame["dataset"].unique()}, runs)
    return build_view(scored, datasets, token_lengths(datasets), runs)


# reading score.py's output


def read_runs(scores_dir: Path) -> list[Run]:
    runs = []
    for meta_path in sorted(scores_dir.rglob("meta.json")):
        csv_path = meta_path.parent / "scores.csv"
        if not csv_path.exists():
            print(f"WARNING: {meta_path} has no scores.csv next to it, skipping")
            continue
        meta: ScoresMeta = json.loads(meta_path.read_text())
        frame, truncated = read_scores_csv(csv_path)
        if truncated:
            print(f"WARNING: {csv_path} ends mid-row (interrupted copy?), dropped the partial last line")
        frame = frame.drop_duplicates(subset=["dataset", "row_id"], keep="last")
        warn_missing_rows(csv_path, meta, frame)
        errored = frame["score"].isna()
        runs.append(Run(meta_path.parent, meta, DetectorConfig(**meta["detector_config"]), frame[~errored],
                        frame[errored], truncated))
    print(f"{len(runs)} run folder(s) under {scores_dir}")
    return runs


def read_scores_csv(path: Path) -> tuple[pd.DataFrame, bool]:
    """Reads scores.csv, dropping a cut-off last line. Returns (frame, truncated)."""
    with path.open("rb") as file:
        file.seek(-1, io.SEEK_END)
        truncated = file.read(1) != b"\n"
    source: Path | io.BytesIO = path
    if truncated:
        # score.py ends rows with \r\n, a newline inside a quoted text is a bare \n
        data = path.read_bytes()
        terminator = b"\r\n" if b"\r\n" in data[:data.find(b"\n") + 1] else b"\n"
        source = io.BytesIO(data[: data.rfind(terminator) + len(terminator)])
    table = pacsv.read_csv(  # ~10x faster than pandas' own parser
        source,
        parse_options=pacsv.ParseOptions(newlines_in_values=True),
        convert_options=pacsv.ConvertOptions(
            include_columns=SCORE_COLUMNS,
            column_types={"row_id": pa.int64(), "label": pa.int8(), "score": pa.float64(), "error": pa.string(),
                          "score_seconds": pa.float64()},
        ),
    )
    frame = table.to_pandas()
    frame[CATEGORY_COLUMNS] = frame[CATEGORY_COLUMNS].astype("category")
    return frame, truncated


def warn_missing_rows(csv_path: Path, meta: ScoresMeta, frame: pd.DataFrame) -> None:
    """Warns if scores.csv has fewer rows than meta.json says were scored."""
    for dataset in {entry["dataset"] for entry in meta["runs"]}:
        expected = max(entry.get("rows_already_done_before_run", 0) + entry.get("rows_scored_this_run", 0)
                       for entry in meta["runs"] if entry["dataset"] == dataset)
        have = int((frame["dataset"] == dataset).sum())
        if have < expected:
            print(f"WARNING: {csv_path} has {have} {dataset} rows, its meta.json says {expected} were scored")


def detector_label(config: DetectorConfig) -> str:
    """e.g. 'fastdetect(Qwen3-0.6B)', 'detectgpt(gpt2, mask=mt5-small)', 'baseline(multitude_en)'."""
    model = (config.scoring or "").split("/")[-1]
    settings = ""
    if config.top_p not in (None, 1.0):
        settings += f", top-p={config.top_p}"
    if config.top_k is not None:
        settings += f", top-k={config.top_k}"
    match config.detector:
        case "baseline":
            return f"baseline({baseline_name(config)})"
        case "fastdetect":
            if config.fastdetect_reference != config.scoring:
                settings = f", ref={(config.fastdetect_reference or '').split('/')[-1]}" + settings
            if config.fastdetect_mode == "sampling":
                settings += f", sampling n={config.fastdetect_n_samples}"
            return f"fastdetect({model}{settings})"
        case "detectgpt":
            return f"detectgpt({model}, mask={(config.detectgpt_mask or '').split('/')[-1]}{settings})"
        case _:
            return config.detector


def method_order(method: str) -> int:
    match method:
        case "fastdetect":
            return 0
        case "detectgpt":
            return 1
        case _:
            return 2


def group_runs(runs: list[Run]) -> dict[Detector, list[Run]]:
    """The runs per detector config, sorted by method, model and label."""
    by_config: dict[DetectorConfig, list[Run]] = {}
    for run in runs:
        by_config.setdefault(run.config, []).append(run)
    labels = [detector_label(config) for config in by_config]
    detectors = [
        Detector(config, label if labels.count(label) == 1 else f"{label} [{str(config)[-16:]}]")
        for config, label in zip(by_config, labels)
    ]

    def order(detector: Detector) -> tuple:
        model = scoring_model(detector.config.scoring)
        return (method_order(detector.method), model.family if model else "", model.params if model else 0,
                detector.label)

    return {detector: by_config[detector.config] for detector in sorted(detectors, key=order)}


def merged_scores(detector: Detector, runs: list[Run]) -> pd.DataFrame:
    """All runs of one detector combined. Warns if two runs disagree on a text."""
    combined = pd.concat([run.scores for run in runs], ignore_index=True)
    if len(runs) > 1:
        spread = combined.groupby(["dataset", "row_id"], observed=True)["score"].agg(["min", "max"])
        conflicting = spread[(spread["max"] - spread["min"]) > 1e-4 * spread["max"].abs().clip(lower=1.0)]
        if len(conflicting):
            print(f"WARNING: {detector.label}: {len(conflicting)} text(s) scored differently across its folders")
    return combined.drop_duplicates(subset=["dataset", "row_id"], keep="last")


def seen_in_training(detector: Detector, dataset: str, splits: pd.Series) -> pd.Series:
    """The rows a detector saw in training. Only baselines have any."""
    if detector.method != "baseline" or baseline_trained_on(detector.config) not in (dataset, "both"):
        return pd.Series(False, index=splits.index)
    return splits != "test"


# the datasets


def load_datasets(names: set[str], runs: list[Run]) -> dict[str, Dataset]:
    """The dataset csvs the scores came from. Warns if a run used a different csv."""
    datasets = {}
    for name in sorted(names & set(DATASET_MODULES)):
        path = DATASET_MODULES[name].DEFAULT_CSV_PATH
        checksum = file_checksum(path)
        datasets[name] = Dataset(DATASET_MODULES[name].load_dataframe(path)[["text", "split"]], checksum)
        for run_dir in sorted({str(run.dir) for run in runs for entry in run.meta["runs"]
                               if entry["dataset"] == name and entry["dataset_csv_sha256"] not in (None, checksum)}):
            print(f"WARNING: {run_dir} scored a different {name} csv than the local one")
    return datasets


def token_lengths(datasets: dict[str, Dataset]) -> dict[str, pd.Series]:
    """Every text's length in mT5 tokens, per dataset (cached on disk)."""
    return {
        name: cached_token_lengths(name, dataset.frame.index.to_series(), dataset.frame["text"], dataset.checksum,
                                   LENGTH_TOKENIZER_MODEL)
        for name, dataset in datasets.items()
    }


# the view


def build_view(
    scored: dict[Detector, pd.DataFrame], datasets: dict[str, Dataset], lengths: dict[str, pd.Series],
    runs: list[Run], subsets: dict[str, pd.Index] | None = None,
) -> View:
    """Each detector's unseen texts per dataset it scored at least MIN_COVERAGE of
    (or has a cut-off scores.csv of). subsets limits this to those row_ids."""
    truncated = {(str(run.config), entry["dataset"]) for run in runs if run.truncated for entry in run.meta["runs"]}
    frames: dict[Detector, dict[str, pd.DataFrame]] = {}
    coverage = []
    for detector, frame in scored.items():
        frames[detector] = {}
        for dataset, rows in frame.groupby("dataset", observed=True):
            dataset = str(dataset)
            if dataset not in datasets or (subsets is not None and dataset not in subsets):
                continue
            texts = datasets[dataset].frame
            expected = texts.index[~seen_in_training(detector, dataset, texts["split"]).to_numpy()]
            rows = rows[~seen_in_training(detector, dataset, rows["split"]).to_numpy()]
            if subsets is not None:
                expected = expected.intersection(subsets[dataset])
                rows = rows[rows["row_id"].isin(subsets[dataset])]
            share = len(rows) / len(expected) if len(expected) else 0.0
            partial = share < MIN_COVERAGE and (detector.key, dataset) in truncated
            included = len(rows) > 0 and (share >= MIN_COVERAGE or partial)
            coverage.append({"detector": detector.key, "label": detector.label, "dataset": dataset,
                             "rows": len(rows), "expected_rows": len(expected), "coverage": share,
                             "included": included, "partial": included and partial})
            if included:
                frames[detector][dataset] = rows.assign(
                    computed_length=lengths[dataset].reindex(rows["row_id"]).to_numpy())
    cells = pd.concat([subgroup_aucs(frame).assign(detector=detector.key, dataset=dataset)
                       for detector, per_dataset in frames.items() for dataset, frame in per_dataset.items()],
                      ignore_index=True)
    return View(frames, cells, pd.DataFrame(coverage))
