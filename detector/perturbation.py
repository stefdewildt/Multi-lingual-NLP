"""

This file contains the perturbation-based (detectgpt) detector: perturb the
text a few times and see how much more likely the original is than its
perturbed neighbors, under a scoring model.

Check out eq 1 of Mireshghallah et al., or DetectGPT / Mitchell et al.'s
Algorithm 1, for details on how this works. Both papers' eq 1 is the same
raw formula. DetectGPT's std-normalized version only appears in Algorithm
1, it isn't given its own numbered equation. Bao et al. also give the 
unnormalized version.

"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from detector.models import Device, forward_logits

Metric = Literal["sum", "average"]
"""Which per-text metric is compared between the original and perturbed
texts. Articles use the text log-likelihood, we propose a second option
which uses the average token log-likelihood.

"sum" uses sum_t log p(token), i.e. what all the articles use.
"average" uses avg_t log p(token), i.e., divides that sum by 
the token count, fixing the sequence-length bias described 
in this folder's README. Actually I just noticed that even
though the FastDetect paper doesn't specify this, they actually
do use the token average in their code."""

CurvatureNormalization = Literal["std", "none"]
"""Whether to divide the curvature by the standard deviation of the
perturbed scores.

"none" is like Mireshghallah et al. and Bao et al..
"std" is like Mitchell et al."""


@dataclass
class PerturbationDetector:
    scoring_model: PreTrainedModel  # model text log-probability is measured under
    scoring_tokenizer: PreTrainedTokenizerBase
    mask_model: PreTrainedModel  # fills masked spans to produce the perturbed neighbor texts
    mask_tokenizer: PreTrainedTokenizerBase
    device: Device
    n_perturbations: int  # how many perturbed neighbors to average the metric over
    metric: Metric  # which per-text score to compare, see Metric
    normalize_by: CurvatureNormalization  # "std" or "none", see CurvatureNormalization
    span_length: int = 2  # tokens per masked span (2 in Mireshghallah et al.)
    pct_masked: float = 0.15  # fraction of the text to mask, spread over spans of span_length
    top_p: float = 1.0  # nucleus sampling cutoff the mask model uses to regenerate a span
    top_k: int | None = None  # top-k cutoff the mask model uses to regenerate a span, None disables it

    def score(self, text: str) -> float:
        """A score indicating the amount of generatedness of the text.

        perturbed_metrics = metric(perturb(text))
        score(text) =
            (metric(text) - mean(perturbed_metrics)
            / std(perturbed_metrics)

        `metric` can be sum or average of token likelihoods.

        Mireshghallah et al. don't divide by std.
        DetectGPT (Mitchell et al.) does.
        """
        original = _text_metric(text, self.scoring_model, self.scoring_tokenizer, self.device, self.metric)
        perturbed_texts = _perturb(
            text,
            self.n_perturbations,
            self.mask_model,
            self.mask_tokenizer,
            self.device,
            self.span_length,
            self.pct_masked,
            self.top_p,
            self.top_k,
        )
        perturbed = [
            _text_metric(t, self.scoring_model, self.scoring_tokenizer, self.device, self.metric)
            for t in perturbed_texts
        ]

        curvature = original - statistics.mean(perturbed)
        if self.normalize_by == "std" and len(perturbed) > 1:
            curvature /= statistics.stdev(perturbed) + 1e-6
        return curvature

    def predict(self, text: str, threshold: float) -> bool:
        return self.score(text) > threshold


def _text_metric(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: Device,
    metric: Metric,
) -> float:
    """The score of generatedness for a single text.

    metric = 1 / |text| * sum_token_in_text log p(token)

    Drops the normalizer if metric=="sum".
    """
    logits, labels = forward_logits(text, model, tokenizer, device)
    lprobs = logits.log_softmax(dim=-1)
    lls = lprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return lls.mean().item() if metric == "average" else lls.sum().item()


MASK_ID = -1  # placeholder id, never a real vocab id (those are all >= 0)
EXTRA_ID_PATTERN = re.compile(r"<extra_id_\d+>")


def _tokenize_and_mask(
    text: str, span_length: int, pct_masked: float, ceil_pct: bool, mask_tokenizer: PreTrainedTokenizerBase
) -> str:
    """Replaces randomly chosen, non-overlapping spans of text with
    sentinel tokens <extra_id_N>.

    These sentinel tokens replace roughly pct_masked of the text,
    in spans of span_length tokens each. Spans are chosen over
    mask_tokenizer's own tokenization of text, not over whitespace-split
    words, so this works the same way for scripts (e.g. Chinese) that
    don't separate words with spaces.

    ceil_pct rounds the number of masked spans up instead of down, e.g. to
    guarantee at least one span on a short text.

    Mireshghallah et al., literature/2305.09859v4.pdf.
    """
    buffer_size = 1  # tokens kept between two masked spans so they don't touch/merge

    token_ids = mask_tokenizer(text, add_special_tokens=False).input_ids
    if len(token_ids) <= span_length:
        return text  # too short to mask

    # number of span_length-token spans needed to mask pct_masked of the text
    n_spans = pct_masked * len(token_ids) / (span_length + buffer_size * 2)
    n_spans = int(np.ceil(n_spans)) if ceil_pct else int(n_spans)

    # repeatedly pick a random span and mask it, skipping ones that would touch an existing mask
    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(token_ids) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(token_ids), end + buffer_size)
        if MASK_ID not in token_ids[search_start:search_end]:
            token_ids[start:end] = [MASK_ID]
            n_masks += 1

    num_filled = 0
    for idx, token_id in enumerate(token_ids):
        if token_id == MASK_ID:
            token_ids[idx] = mask_tokenizer.convert_tokens_to_ids(f"<extra_id_{num_filled}>")
            num_filled += 1
    return cast(str, mask_tokenizer.decode(token_ids, skip_special_tokens=False))


def _count_masks(texts: list[str]) -> list[int]:
    return [len(EXTRA_ID_PATTERN.findall(text)) for text in texts]


def _replace_masks(
    texts: list[str],
    mask_model: PreTrainedModel,
    mask_tokenizer: PreTrainedTokenizerBase,
    device: Device,
    top_p: float,
    top_k: int | None,
) -> list[str]:
    n_expected = _count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(device)
    outputs = cast(Any, mask_model).generate(
        **tokens,
        max_length=150,
        do_sample=True,
        top_p=top_p,
        top_k=top_k if top_k is not None else 0,  # 0 disables top-k filtering in generate()
        num_return_sequences=1,
        eos_token_id=stop_id,
    )
    return mask_tokenizer.batch_decode(cast(Tensor, outputs), skip_special_tokens=False)


def _extract_fills(texts: list[str]) -> list[list[str]]:
    texts = [text.replace("<pad>", "").replace("</s>", "").strip() for text in texts]
    fills = [EXTRA_ID_PATTERN.split(text)[1:-1] for text in texts]
    return [[fill.strip() for fill in text_fills] for text_fills in fills]


def _apply_extracted_fills(masked_texts: list[str], extracted_fills: list[list[str]]) -> list[str]:
    n_expected = _count_masks(masked_texts)

    results = []
    for masked_text, fills, n in zip(masked_texts, extracted_fills, n_expected):
        if len(fills) < n:
            results.append("")
            continue
        # chunks are the pieces of masked_text around each <extra_id_N>, in
        # order. interleaving them with the fills splices the generated
        # content back in without assuming words are space-separated.
        chunks = EXTRA_ID_PATTERN.split(masked_text)
        pieces = [chunks[0]]
        for fill, chunk in zip(fills, chunks[1:]):
            pieces += [fill, chunk]
        results.append("".join(pieces))

    return results

def _perturb(
    text: str,
    n: int,
    mask_model: PreTrainedModel,
    mask_tokenizer: PreTrainedTokenizerBase,
    device: Device,
    span_length: int,
    pct_masked: float,
    top_p: float,
    top_k: int | None,
) -> list[str]:
    """Masks n independent copies of text (tokenize_and_mask), fills the
    masks with mask_model (_replace_masks), then extracts and splices the
    fills back in (_extract_fills, _apply_extracted_fills).

    Retries any copy whose fill comes back empty, with a fresh mask, until
    every copy has text.
    """
    def fill(masked_texts: list[str]) -> list[str]:
        raw_fills = _replace_masks(masked_texts, mask_model, mask_tokenizer, device, top_p, top_k)
        extracted_fills = _extract_fills(raw_fills)
        return _apply_extracted_fills(masked_texts, extracted_fills)

    masked_texts = [_tokenize_and_mask(text, span_length, pct_masked, True, mask_tokenizer) for _ in range(n)]
    perturbed_texts = fill(masked_texts)

    while "" in perturbed_texts:
        empty = [idx for idx, t in enumerate(perturbed_texts) if t == ""]
        retried = fill([_tokenize_and_mask(text, span_length, pct_masked, True, mask_tokenizer) for _ in empty])
        for idx, t in zip(empty, retried):
            perturbed_texts[idx] = t

    return perturbed_texts
