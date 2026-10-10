"""Train the TF-IDF baselines on the training splits of MultiSocial and
MULTITuDE combined, for the same language sets as the single-dataset ones
(todo.md): all, en, big_eu, less_common. Saved to
detector/baseline/models/both_<set>. Score them with score.py on the test
splits only (the training splits were seen in training)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from detector.baseline.baseline_detector import DATASET_CLASSES, BaselineDetector, _frame, _sample_balanced

LANGUAGE_SETS = {
    "all": None,
    "en": ["en"],
    "big_eu": ["fr", "de", "en", "es", "pt"],
    "less_common": ["nl", "et", "ga", "gd"],
}


def train_both(name: str, languages: list[str] | None) -> None:
    frames = [_frame(DATASET_CLASSES[key](split="train", languages=languages)) for key in ("multisocial", "multitude")]
    frame = _sample_balanced(pd.concat(frames, ignore_index=True), None, balance_classes=True,
                             balance_languages=False, balance_models=False)
    detector = BaselineDetector(max_features=5_000, ngram_range=(1, 2))
    detector.fit(frame["text"], frame["label"])
    detector.save(Path(f"detector/baseline/models/both_{name}"))
    print(f"both_{name}: trained on {len(frame)} samples", flush=True)


if __name__ == "__main__":
    for name, languages in LANGUAGE_SETS.items():
        train_both(name, languages)
