from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from transformers import PreTrainedTokenizerBase

from detector.models import load_tokenizer

LengthUnit = Literal["words", "chars", "tokens"]

LENGTH_TOKENIZER_MODEL = "google/mt5-small"
"""The model of which we use it's tokenizer to find the length of texts. 
Should be multilingual so it works well on all languages, and should be 
relatively normalized between languages (every token holds roughly the 
same amount of information, no matter the language)."""

CACHE_PATH = Path("outputs/cache/token_lengths.csv")
CACHE_COLUMNS = ["dataset", "row_id", "tokenizer_model", "dataset_csv_sha256", "token_length"]
"""We keep a cache file for text lengths so we don't have to load and 
tokenize everything all the time."""


def text_length(text: str, unit: LengthUnit, tokenizer: PreTrainedTokenizerBase | None) -> int:
    if unit == "words":
        return len(text.split())
    if unit == "chars":
        return len(text)
    assert tokenizer is not None, "tokenizer is required for unit='tokens'"
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def cached_token_lengths(
    dataset: str,
    row_ids: pd.Series,
    texts: pd.Series,
    dataset_csv_sha256: str | None,
    tokenizer_model: str,
    cache_path: Path = CACHE_PATH,
) -> pd.Series:
    """Token length for each of `texts`. Rows already cached for this 
    (dataset, tokenizer_model) with a matching dataset_csv_sha256 are 
    reused. Rverything else is tokenized and the cache file is updated."""
    cache = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=CACHE_COLUMNS)
    own = pd.DataFrame({"dataset": dataset, "row_id": row_ids.to_numpy(), "text": texts.to_numpy()})
    subset = cache[(cache["dataset"] == dataset) & (cache["tokenizer_model"] == tokenizer_model)]
    merged = own.merge(subset[["row_id", "dataset_csv_sha256", "token_length"]], on="row_id", how="left")
    stale = merged["token_length"].isna() | (merged["dataset_csv_sha256"] != dataset_csv_sha256)

    if stale.any():
        tokenizer = load_tokenizer(tokenizer_model)
        merged.loc[stale, "token_length"] = merged.loc[stale, "text"].map(
            lambda t: text_length(str(t), "tokens", tokenizer)
        )
        new_rows = merged.loc[stale, ["row_id", "token_length"]].assign(
            dataset=dataset, tokenizer_model=tokenizer_model, dataset_csv_sha256=dataset_csv_sha256
        )[CACHE_COLUMNS]
        cache = pd.concat([cache, new_rows], ignore_index=True).drop_duplicates(
            subset=["dataset", "row_id", "tokenizer_model"], keep="last"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(cache_path, index=False)

    return pd.Series(merged["token_length"].to_numpy(), index=texts.index)
