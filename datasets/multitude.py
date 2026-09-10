"""

This file should contain some data loading utilities, for loading and using data from the MULTITuDE dataset.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal, TypedDict, cast

import pandas as pd
from torch.utils.data import Dataset

DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "MULTITuDE" / "multitude.csv"

Split = Literal["train", "test"]
"""The 'split' column of the dataset."""

Language = Literal["en", "es", "ru", "nl", "ca", "cs", "de", "zh", "pt", "ar", "uk"]
"""The 'language' column of the dataset (ISO 639-1 codes)."""

GeneratorModel = Literal[
    "gpt-3.5-turbo",
    "gpt-4",
    "text-davinci-003",
    "alpaca-lora-30b",
    "vicuna-13b",
    "opt-66b",
    "llama-65b",
    "opt-iml-max-1.3b",
    "human",
]
"""The 'multi_label' column of the dataset: the model that generated the
text, or "human" if the text is human-written."""


class MultitudeSample(TypedDict):
    """A single, typed row of the MULTITuDE dataset."""

    text: str
    label: int  # 0 = human-written, 1 = machine-generated
    multi_label: GeneratorModel
    split: Split
    language: Language
    length: int
    source: str


def load_dataframe(csv_path: str | Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Read the raw multitude.csv into a DataFrame.

    Raises a helpful error if the csv hasn't been downloaded yet, see
    datasets/MULTITuDE/README.md.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Download multitude.csv from "
            "https://zenodo.org/records/10013755 and place it there "
            "(see datasets/MULTITuDE/README.md)."
        )
    return pd.read_csv(csv_path)


def filter_dataframe(
    df: pd.DataFrame,
    split: Split | str | None = None,
    languages: Language | str | Iterable[Language | str] | None = None,
    models: GeneratorModel | str | Iterable[GeneratorModel | str] | None = None,
    binary_label: int | None = None,
) -> pd.DataFrame:
    """Filter the dataframe on split/language/generator-model/binary label.

    Any of the filters left as None are not applied. `languages` and
    `models` accept either a single value or an iterable of values.
    """

    def _as_str_list(value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        return list(value)

    mask = pd.Series(True, index=df.index)

    if split is not None:
        mask &= df["split"] == split

    language_values = _as_str_list(languages)
    if language_values is not None:
        mask &= df["language"].isin(language_values)

    model_values = _as_str_list(models)
    if model_values is not None:
        mask &= df["multi_label"].isin(model_values)

    if binary_label is not None:
        mask &= df["label"] == binary_label

    return cast(pd.DataFrame, df.loc[mask])


def row_to_sample(row: Any) -> MultitudeSample:
    """Convert a raw DataFrame row into a typed MultitudeSample."""
    return MultitudeSample(
        text=row["text"],
        label=int(row["label"]),
        multi_label=row["multi_label"],
        split=row["split"],
        language=row["language"],
        length=int(row["length"]),
        source=row["source"],
    )


class MultitudeDataset(Dataset):
    """PyTorch Dataset over the MULTITuDE dataset.

    Each item is a `MultitudeSample` dict, which is also batchable by the
    default DataLoader collate function.
    """

    def __init__(
        self,
        csv_path: str | Path = DEFAULT_CSV_PATH,
        split: Split | str | None = None,
        languages: Language | str | Iterable[Language | str] | None = None,
        models: GeneratorModel | str | Iterable[GeneratorModel | str] | None = None,
        binary_label: int | None = None,
        dataframe: pd.DataFrame | None = None,
    ) -> None:
        df = dataframe if dataframe is not None else load_dataframe(csv_path)
        df = filter_dataframe(
            df, split=split, languages=languages, models=models, binary_label=binary_label
        )
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> MultitudeSample:
        return row_to_sample(self.df.iloc[idx])
