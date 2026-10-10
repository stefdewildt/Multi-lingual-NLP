"""What we know about the detectors and the text languages: which scoring
model a detector uses, which detectors are in the analysis, what a baseline
was trained on, what the scoring and reference models were trained on, etc,
and how well resourced each text language is."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PureWindowsPath

from score import DetectorConfig

ENGLISH = frozenset({"en"})
LLAMA_3_2 = frozenset({"en", "de", "fr", "it", "pt", "hi", "es", "th"})  # model card: officially supported
EUROLLM = frozenset({  # model card: 24 EU languages + 11 others
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "ga", "it", "lv", "lt", "mt", "pl", "pt",
    "ro", "sk", "sl", "es", "sv", "ar", "ca", "zh", "gl", "hi", "ja", "ko", "no", "ru", "tr", "uk",
})
QWEN3 = frozenset({  # 119 languages, of ours all but Gaelic, which the model card doesn't confirm
    "ar", "bg", "ca", "cs", "de", "el", "en", "es", "et", "hr", "hu", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "uk",
    "zh",
})


@dataclass(frozen=True)
class ScoringModel:
    family: str  # e.g. "Qwen"
    params: float  # billions of parameters
    training_languages: frozenset[str]  # officially in the training data


def scoring_model(name: str | None) -> ScoringModel | None:
    """The scoring models in the analysis, None for any other (test or scouting) model."""
    match name:
        case "gpt2":
            return ScoringModel("GPT-2", 0.124, ENGLISH)
        case "gpt2-xl":
            return ScoringModel("GPT-2", 1.5, ENGLISH)
        case "HuggingFaceTB/SmolLM2-360M":
            return ScoringModel("SmolLM2", 0.36, ENGLISH)
        case "HuggingFaceTB/SmolLM2-1.7B":
            return ScoringModel("SmolLM2", 1.7, ENGLISH)
        case "Qwen/Qwen3-0.6B":
            return ScoringModel("Qwen", 0.6, QWEN3)
        case "Qwen/Qwen3-1.7B":
            return ScoringModel("Qwen", 1.7, QWEN3)
        case "Qwen/Qwen3-4B":
            return ScoringModel("Qwen", 4.0, QWEN3)
        case "meta-llama/Llama-3.2-1B-Instruct":
            return ScoringModel("Llama", 1.2, LLAMA_3_2)
        case "meta-llama/Llama-3.2-3B-Instruct":
            return ScoringModel("Llama", 3.2, LLAMA_3_2)
        case "utter-project/EuroLLM-1.7B":
            return ScoringModel("EuroLLM", 1.7, EUROLLM)
        case _:
            return None


SUPERSEDED_BASELINES = frozenset({"b0d85f30", "ff7254fa", "e51d4d77", "e42bd423"})
"""Weight hashes of the first MultiSocial baselines, replaced by the set that
was trained together with the MULTITuDE baselines."""


def baseline_name(config: DetectorConfig) -> str:
    """The name of a baseline's weights folder, e.g. 'multitude_en'."""
    return PureWindowsPath(config.baseline_weights_path or "").name


def baseline_trained_on(config: DetectorConfig) -> str:
    """The dataset(s) a baseline was trained on: "multisocial", "multitude" or "both"."""
    return baseline_name(config).split("_", 1)[0]


def baseline_languages(config: DetectorConfig) -> str:
    """The languages a baseline was trained on: "all", "en", "big_eu" or "less_common"."""
    return baseline_name(config).split("_", 1)[1]


def baseline_training_languages(config: DetectorConfig) -> frozenset[str] | None:
    """The languages a baseline was trained on, None if it was trained on all of them."""
    match baseline_languages(config):
        case "en":
            return frozenset({"en"})
        case "big_eu":
            return frozenset({"fr", "de", "en", "es", "pt"})
        case "less_common":
            return frozenset({"nl", "et", "ga", "gd"})
        case _:
            return None


def role(config: DetectorConfig) -> str:
    """main (in the analysis), tuning (a hyperparameter variant) or excluded."""
    match config.detector:
        case "baseline":
            superseded = (config.baseline_weights_hash or "")[:8] in SUPERSEDED_BASELINES
            if baseline_trained_on(config) not in ("multisocial", "multitude", "both") or superseded:
                return "excluded"
            return "main"
        case "detectgpt":
            if scoring_model(config.scoring) is None or config.detectgpt_mask != "google/mt5-small":
                return "excluded"
            return "main" if config.top_p in (None, 1.0) and config.top_k is None else "tuning"
        case "fastdetect":
            if scoring_model(config.scoring) is None:
                return "excluded"
            default = (config.top_p in (None, 1.0) and config.top_k is None and config.fastdetect_mode == "analytic"
                       and config.fastdetect_reference == config.scoring)
            return "main" if default else "tuning"
        case _:
            return "excluded"


RESOURCE_LEVELS = ["high", "upper-mid", "mid", "low"]


def resource_level(language: str) -> str:
    """Resource level from Joshi et al. (2020): high = class 5, upper-mid = 4, mid = 3, low = 0-2."""
    match language:
        case "en" | "de" | "es" | "fr" | "ar" | "zh":
            return "high"
        case "pt" | "nl" | "ca" | "cs" | "hr" | "hu" | "pl" | "ru":
            return "upper-mid"
        case "ro" | "et" | "bg" | "el" | "sk" | "sl" | "uk":
            return "mid"
        case _:  # ga (class 2), gd (class 1)
            return "low"


def by_resource_level(languages) -> list[str]:
    """Languages from high to low resource, alphabetical within a level."""
    return sorted(set(languages), key=lambda language: (RESOURCE_LEVELS.index(resource_level(language)), language))
