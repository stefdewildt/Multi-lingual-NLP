"""

This file should contain some data loading utilities, for loading and using data from the MULTITuDE dataset.

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, cast

import pandas as pd
from torch.utils.data import DataLoader, Dataset

DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "MULTITuDE" / "multitude.csv"


class Split(str, Enum):
    """The 'split' column of the dataset."""

    TRAIN = "train"
    TEST = "test"


class Language(str, Enum):
    """The 'language' column of the dataset (ISO 639-1 codes)."""

    ENGLISH = "en"
    SPANISH = "es"
    RUSSIAN = "ru"
    DUTCH = "nl"
    CATALAN = "ca"
    CZECH = "cs"
    GERMAN = "de"
    CHINESE = "zh"
    PORTUGUESE = "pt"
    ARABIC = "ar"
    UKRAINIAN = "uk"


class GeneratorModel(str, Enum):
    """The 'multi_label' column of the dataset: the model that generated the
    text, or HUMAN if the text is human-written."""

    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    TEXT_DAVINCI_003 = "text-davinci-003"
    ALPACA_LORA_30B = "alpaca-lora-30b"
    VICUNA_13B = "vicuna-13b"
    OPT_66B = "opt-66b"
    LLAMA_65B = "llama-65b"
    OPT_IML_MAX_1_3B = "opt-iml-max-1.3b"
    HUMAN = "human"


@dataclass(frozen=True)
class MultitudeSample:
    """A single, typed row of the MULTITuDE dataset."""

    text: str
    label: int  # 0 = human-written, 1 = machine-generated
    multi_label: GeneratorModel
    split: Split
    language: Language
    length: int
    source: str

    @property
    def is_machine_generated(self) -> bool:
        return self.label == 1


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
        if isinstance(value, (str, Enum)):
            return [str(value.value if isinstance(value, Enum) else value)]
        return [str(v.value if isinstance(v, Enum) else v) for v in value]

    mask = pd.Series(True, index=df.index)

    if split is not None:
        split_value = split.value if isinstance(split, Enum) else split
        mask &= df["split"] == split_value

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
        multi_label=GeneratorModel(row["multi_label"]),
        split=Split(row["split"]),
        language=Language(row["language"]),
        length=int(row["length"]),
        source=row["source"],
    )


def load_samples(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    split: Split | str | None = None,
    languages: Language | str | Iterable[Language | str] | None = None,
    models: GeneratorModel | str | Iterable[GeneratorModel | str] | None = None,
    binary_label: int | None = None,
) -> list[MultitudeSample]:
    """Load the dataset (optionally filtered) as a list of typed samples."""
    df = load_dataframe(csv_path)
    df = filter_dataframe(
        df, split=split, languages=languages, models=models, binary_label=binary_label
    )
    return [row_to_sample(row) for _, row in df.iterrows()]


class MultitudeDataset(Dataset):
    """PyTorch Dataset over the MULTITuDE dataset.

    Each item is a plain dict (rather than MultitudeSample) so that it can
    be batched by the default DataLoader collate function. Use
    `get_sample` if you want the typed dataclass for a given index.
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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "text": row["text"],
            "label": int(row["label"]),
            "multi_label": str(row["multi_label"]),
            "split": str(row["split"]),
            "language": str(row["language"]),
            "length": int(row["length"]),
            "source": str(row["source"]),
        }

    def get_sample(self, idx: int) -> MultitudeSample:
        return row_to_sample(self.df.iloc[idx])


def make_dataloader(
    dataset: MultitudeDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    **kwargs: Any,
) -> DataLoader:
    """Thin convenience wrapper around DataLoader with sensible defaults."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
