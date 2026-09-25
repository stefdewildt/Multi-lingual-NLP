"""

This file contains some data loading utilities, for loading and using data from the MultiSocial dataset.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal, TypedDict, cast

import pandas as pd
from torch.utils.data import Dataset

DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "MultiSocial" / "multisocial.csv"

Split = Literal["train", "test"]
"""The 'split' column of the dataset."""

Language = Literal[
    "ar", "bg", "ca", "cs", "de", "el", "en", "es", "et", "ga", "gd", "hr",
    "hu", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "uk", "zh",
]
"""The 'language' column of the dataset (ISO 639-1 codes)."""

GeneratorModel = Literal[
    "Mistral-7B-Instruct-v0.2",
    "aya-101",
    "gemini",
    "gpt-3.5-turbo-0125",
    "opt-iml-max-30b",
    "v5-Eagle-7B-HF",
    "vicuna-13b",
    "human",
]
"""The 'multi_label' column of the dataset: the model that generated the
text, or "human" if the text is human-written."""

Source = Literal[
    "discord", "gab", "telegram", "twitter", "whatsapp",
    "multisocial_discord", "multisocial_gab", "multisocial_telegram",
    "multisocial_twitter", "multisocial_whatsapp",
]
"""The 'source' column of the dataset. Some values carry a 'multisocial_'
prefix and some don't, this is a quirk of the raw data, not a typo."""


class MultiSocialSample(TypedDict):
    """A single, typed row of the MultiSocial dataset."""

    text: str
    label: int  # 0 = human-written, 1 = machine-generated
    multi_label: GeneratorModel
    split: Split
    language: Language
    length: int
    source: Source
    potential_noise: int  # 0 = no identified noise, 1 = potential noise


def load_dataframe(csv_path: str | Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Read the raw multisocial.csv into a DataFrame.

    Raises a helpful error if the csv hasn't been downloaded yet, see
    datasets/MultiSocial/README.md.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Download multisocial.csv from "
            "https://zenodo.org/records/13846152 and place it there "
            "(see datasets/MultiSocial/README.md)."
        )
    with csv_path.open("rb") as file:
        if file.read(2) == b"\x1f\x8b":
            raise ValueError(f"{csv_path} is a gzip file, unzip it first, e.g. gunzip {csv_path}")
    return pd.read_csv(csv_path)


def filter_dataframe(
    df: pd.DataFrame,
    split: Split | str | None = None,
    languages: Language | str | Iterable[Language | str] | None = None,
    models: GeneratorModel | str | Iterable[GeneratorModel | str] | None = None,
    binary_label: int | None = None,
    exclude_noise: bool = False,
) -> pd.DataFrame:
    """Filter the dataframe on split/language/generator-model/binary label.

    Any of the filters left as None are not applied. `languages` and
    `models` accept either a single value or an iterable of values.
    `exclude_noise`, if True, drops rows with potential_noise == 1.
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

    if exclude_noise:
        mask &= df["potential_noise"] == 0

    return cast(pd.DataFrame, df.loc[mask])


def row_to_sample(row: Any) -> MultiSocialSample:
    """Convert a raw DataFrame row into a typed MultiSocialSample."""
    return MultiSocialSample(
        text=row["text"],
        label=int(row["label"]),
        multi_label=row["multi_label"],
        split=row["split"],
        language=row["language"],
        length=int(row["length"]),
        source=row["source"],
        potential_noise=int(row["potential_noise"]),
    )


class MultiSocialDataset(Dataset):
    """PyTorch Dataset over the MultiSocial dataset.

    Each item is a `MultiSocialSample` dict, which is also batchable by the
    default DataLoader collate function.
    """

    def __init__(
        self,
        csv_path: str | Path = DEFAULT_CSV_PATH,
        split: Split | str | None = None,
        languages: Language | str | Iterable[Language | str] | None = None,
        models: GeneratorModel | str | Iterable[GeneratorModel | str] | None = None,
        binary_label: int | None = None,
        exclude_noise: bool = False,
        dataframe: pd.DataFrame | None = None,
    ) -> None:
        df = dataframe if dataframe is not None else load_dataframe(csv_path)
        df = filter_dataframe(
            df,
            split=split,
            languages=languages,
            models=models,
            binary_label=binary_label,
            exclude_noise=exclude_noise,
        )
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> MultiSocialSample:
        return row_to_sample(self.df.iloc[idx])
