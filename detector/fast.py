"""

This file contains the fast-detectgpt detector: conditional probability
curvature between a reference model and a scoring model, computed either
analytically or by resampling.

Check out eq 3 and eq 4 of Bao et al., Fast-DetectGPT (2024,
literature/2310.05130v3.pdf) for the formula this implements.

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from detector.models import Device, forward_logits

Mode = Literal["analytic", "sampling"]
"""Which estimator computes the conditional probability curvature.

"analytic" is the closed-form estimator, conditional_curvature_analytic:
exact, computed directly from the full softmax distribution.

"sampling" is the Monte Carlo estimator, conditional_curvature_sampling:
approximate, estimated from resampled tokens."""


@dataclass
class FastDetector:
    reference_model: PreTrainedModel  # samples/scores the conditional distribution text is compared against
    reference_tokenizer: PreTrainedTokenizerBase
    scoring_model: PreTrainedModel  # scores the actual observed text
    scoring_tokenizer: PreTrainedTokenizerBase
    mode: Mode  # "analytic" (closed form) or "sampling" (Monte Carlo)
    n_samples: int | None  # required when mode is "sampling", unused (pass None) when "analytic"
    sample_top_p: float | None  # nucleus cutoff for sampling mode, None samples from the full distribution
    sample_top_k: int | None  # top-k cutoff for sampling mode, None disables it
    device: Device

    def score(self, text: str) -> float:
        """A score indicating the amount of generatedness of the text.

        d(x, p_score, p_ref) =
            (log p_score(x) - E_{p_ref}[log p_score(x')])
            / std(log p_score(x'))

        where

        p_ref(x' | x) = prod_t p_ref_token(x'_t | x_<t)
        p_score(x' | x) = prod_t p_score_token(x'_t | x_<t)

        x is the input text, and x' a neighbour text sampled from 
        a conditional reference model. x'_t is the token of x' at
        position t. x_<t are all tokens if x before position t.

        Bao et al. 2024 (literature/2310.05130v3.pdf), eq. 3 and eq. 4.
        The expectation/std are computed exactly if mode is "analytic",
        or estimated by resampling if mode is "sampling".
        """
        logits_score, labels = forward_logits(
            text, self.scoring_model, self.scoring_tokenizer, self.device
        )
        if self.reference_model is self.scoring_model:
            logits_ref = logits_score
        else:
            logits_ref, labels_ref = forward_logits(
                text, self.reference_model, self.reference_tokenizer, self.device
            )
            assert torch.equal(
                labels, labels_ref
            ), "reference and scoring tokenizer disagree"
        if self.mode == "analytic":
            return _conditional_curvature_analytic(logits_ref, logits_score, labels)
        if self.mode == "sampling":
            assert self.n_samples is not None, "n_samples is required when mode is 'sampling'"
            return _conditional_curvature_sampling(
                logits_ref, logits_score, labels, self.n_samples, self.sample_top_p, self.sample_top_k
            )
        raise KeyError(f"unknown mode {self.mode!r}, expected 'analytic' or 'sampling'")

    def predict(self, text: str, threshold: float) -> bool:
        return self.score(text) > threshold


def _filter_logits(logits: Tensor, top_k: int | None, top_p: float | None) -> Tensor:
    """Sets logits outside the top_k highest, and/or outside the smallest
    top_p cumulative probability mass, to -inf.

    Either filter is skipped when its argument is None.

    logits: (..., V). Returns the same shape.
    """
    logits = logits.clone()
    if top_k is not None:
        kth_value = logits.topk(top_k, dim=-1).values[..., -1, None]
        logits[logits < kth_value] = float("-inf")
    if top_p is not None:
        sorted_logits, sorted_index = logits.sort(dim=-1, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = sorted_remove.scatter(-1, sorted_index, sorted_remove)
        logits[remove] = float("-inf")
    return logits


def _sample(logits_ref: Tensor, n_samples: int, top_p: float | None, top_k: int | None) -> Tensor:
    """Draws token ids per position from logits_ref's own distribution.

    x'_t ~ Categorical(softmax(logits_ref_t))

    Restricting top_k/top_p changes the result, so it's no longer
    comparable to conditional_curvature_analytic, which always uses
    the full distribution. Use top_p=None, top_k=None to match it.
    The FastDetect article itself doesn't use top_k or top_p, but
    perhaps it helps?

    logits_ref: (1, T, V). Returns (1, T, n_samples).
    """
    logits_ref = _filter_logits(logits_ref, top_k, top_p)
    lprobs = logits_ref.log_softmax(dim=-1)
    distribution = torch.distributions.Categorical(logits=lprobs)
    return distribution.sample([n_samples]).permute(1, 2, 0)


def _match_vocab(logits_ref: Tensor, logits_score: Tensor) -> tuple[Tensor, Tensor]:
    """Truncates both logits to the smaller vocab size, so the two models
    can be compared when their tokenizers have different vocab sizes.

    This is how fast-detect-gpt has implemented it. Though it's a bit
    problematic since this assumes both vocabulairies are comparable
    and have the same ordering. So this should only be used when you are 
    absolutely sure that their vocabs are the same, but one of them is
    extended in some way. 

    logits_ref, logits_score: (1, T, V_ref) and (1, T, V_score).
    Returns both truncated to (1, T, min(V_ref, V_score)).
    """
    vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
    if logits_ref.size(-1) != logits_score.size(-1):
        warnings.warn(
            f"reference and scoring model vocab sizes differ ({logits_ref.size(-1)} vs "
            f"{logits_score.size(-1)}); truncating to {vocab_size} assumes their vocabularies "
            "agree on token ids up to that point, which is only true if one is an extended "
            "version of the other."
        )
    return logits_ref[..., :vocab_size], logits_score[..., :vocab_size]


def _conditional_curvature_analytic(
    logits_ref: Tensor, logits_score: Tensor, labels: Tensor
) -> float:
    """Since we have access to the reference model p_ref, we
    don't have to use sampling methods to calculate the 
    expectations and variances. We can find them analytically.

    Bao et al. 2024 (literature/2310.05130v3.pdf), eq. 3 and eq. 4.

    logits_ref, logits_score: (1, T, V), labels: (1, T).
    """
    logits_ref, logits_score = _match_vocab(logits_ref, logits_score)
    lprobs_score = logits_score.log_softmax(dim=-1)
    probs_ref = logits_ref.softmax(dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * lprobs_score.square()).sum(dim=-1) - mean_ref.square()
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / (
        var_ref.sum(dim=-1).sqrt() + 1e-6
    )
    return discrepancy.mean().item()


def _gather_log_likelihood(logits: Tensor, index: Tensor) -> Tensor:
    """Per-token log-likelihood under logits at the given token ids,
    averaged over the sequence.

    index is either labels (1, T) or resampled tokens (1, T, n_samples),
    returning (1, 1) or (1, n_samples) respectively.
    """
    index = index.unsqueeze(-1) if index.ndim == logits.ndim - 1 else index
    lprobs = logits.log_softmax(dim=-1)
    return lprobs.gather(dim=-1, index=index).mean(dim=1)


def _conditional_curvature_sampling(
    logits_ref: Tensor, logits_score: Tensor, labels: Tensor, n_samples: int, top_p: float | None, top_k: int | None
) -> float:
    """Even though we can find the analytical solution for the discrepancy,
    we could also do it using sampling methods. Either to check if both
    methods return the same results, or because a different top_k or top_p
    might help us.

    Bao et al. 2024 (literature/2310.05130v3.pdf), eq. 3 and eq. 4.

    logits_ref, logits_score: (1, T, V), labels: (1, T).
    """
    logits_ref, logits_score = _match_vocab(logits_ref, logits_score)
    samples = _sample(logits_ref, n_samples, top_p, top_k)
    log_likelihood_x = _gather_log_likelihood(logits_score, labels)
    log_likelihood_x_tilde = _gather_log_likelihood(logits_score, samples)
    discrepancy = (
        log_likelihood_x.squeeze(-1) - log_likelihood_x_tilde.mean(dim=-1)
    ) / log_likelihood_x_tilde.std(dim=-1)
    return discrepancy.item()
