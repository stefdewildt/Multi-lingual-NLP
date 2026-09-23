"""

This file contains utilities for loading pretrained huggingface models and
tokenizers, and running a forward pass to get next-token logits.

"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

Device = Literal["cpu", "cuda", "mps"] | torch.device | str
"""Anything torch.Tensor.to()/nn.Module.to() accepts."""

ModelKind = Literal["causal", "seq2seq"]
"""Whether a model is a decoder-only causal LM or an encoder-decoder seq2seq model."""

ModelName = Literal[
    # used by fast-detectgpt
    "gpt2",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-j-6B",
    "tiiuae/falcon-7b",
    "tiiuae/falcon-7b-instruct",
    "sberbank-ai/mGPT",
    # used by the perturbation-based detectgpt paper (mask-filling models)
    "t5-small",
    "t5-large",
    "t5-3b",
    "google/mt5-small",
    "google/mt5-large",
    "google/mt5-xl",
    # small, recent, multilingual baselines
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "facebook/xglm-564M",
    "bigscience/bloom-560m",
    # small, recent, monolingual/paucilingual baselines (one or few languages)
    "uer/gpt2-chinese-cluecorpussmall",
    "dbmdz/german-gpt2",
    "DeepESP/gpt2-spanish",
    # tiny models for local testing
    "sshleifer/tiny-gpt2",
    "hf-internal-testing/tiny-random-t5",
]
"""Some known model ids, for convenience/autocomplete. Any huggingface model
id works too, load_model/load_tokenizer just forward it to transformers."""


def load_tokenizer(
    model_name: ModelName | str, cache_dir: str | Path | None = None
) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)


def load_model(
    model_name: ModelName | str,
    kind: ModelKind,
    device: Device,
    cache_dir: str | Path | None = None,
) -> PreTrainedModel:
    model_cls = AutoModelForCausalLM if kind == "causal" else AutoModelForSeq2SeqLM
    model = model_cls.from_pretrained(model_name, cache_dir=cache_dir)
    model.eval()  # disables dropout, torch.no_grad() alone does not do this.
    return model.to(device)  # type: ignore[arg-type]


def forward_logits(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: Device,
) -> tuple[Tensor, Tensor]:
    """Runs a forward pass of text on huggingface model.

    logits[:, i] are the model's predicted next-token logits after
    seeing labels[:, :i], shifted for next-token prediction

    Returns (logits, labels), both shape (1, n_tokens - 1).
    """
    tokenized = tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(device)
    with torch.no_grad():
        logits = model(**tokenized).logits[:, :-1]
    labels = tokenized.input_ids[:, 1:]
    return logits, labels
