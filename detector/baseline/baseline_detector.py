"""TF-IDF and logistic-regression baseline for AI-generated text detection.

The baseline represents each document as a sparse vector of word n-gram
TF-IDF scores and uses logistic regression to predict a binary label:

* ``0`` means human-written text.
* ``1`` means machine-generated text.

The :func:`train_detector` function connects the model to the project's
MultiSocial and MULTITuDE dataset loaders. Evaluation is handled separately
by :func:`evaluate_detector`, which can also save misclassified samples for
error analysis.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from datasets.multisocial import MultiSocialDataset
from datasets.multitude import MultitudeDataset


DATASET_CLASSES = {
    "multisocial": MultiSocialDataset,
    "multitude": MultitudeDataset,
}


class BaselineDetector:
    """Binary TF-IDF classifier where 0 is human and 1 is machine-generated.

    The vectorizer learns the vocabulary and inverse-document frequencies from
    training text. The logistic-regression model then learns how useful those
    features are for separating the two classes. Keeping these objects
    together is important because prediction must use the same vocabulary that
    was learned during training.
    """

    def __init__(
        self,
        max_features: int = 5_000,
        ngram_range: tuple[int, int] = (1, 2),
        class_weight: str | None = "balanced",
    ) -> None:
        """Create an untrained detector.

        Args:
            max_features: Maximum number of vocabulary entries retained by
                the TF-IDF vectorizer.
            ngram_range: Smallest and largest n-gram sizes. ``(1, 2)`` uses
                both individual words and adjacent word pairs.
            class_weight: Logistic-regression weighting strategy. ``"balanced"``
                gives more influence to the minority class during training.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            lowercase=True,
        )
        self.model = LogisticRegression(
            class_weight=class_weight,
            max_iter=1_000,
            random_state=42,
        )
        self.is_trained = False

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> "BaselineDetector":
        """Fit the vectorizer and classifier.

        TF-IDF is fitted here, rather than separately at prediction time, so
        every later input is represented in the same feature space. The input
        labels must contain both binary classes.

        Args:
            texts: Training documents.
            labels: Matching labels using 0 for human and 1 for machine text.

        Returns:
            This detector, allowing ``BaselineDetector().fit(texts, labels)``.
        """
        text_list = list(texts)
        label_array = np.asarray(list(labels), dtype=int)
        if len(text_list) != len(label_array):
            raise ValueError("texts and labels must have the same length")
        if len(np.unique(label_array)) < 2:
            raise ValueError("training data must contain both classes")

        features = self.vectorizer.fit_transform(text_list)
        self.model.fit(features, label_array)
        self.is_trained = True
        
        return self

    def predict(self, text: str) -> tuple[int, float]:
        """Classify one document.

        Returns:
            A pair containing the predicted label and the model probability of
            that predicted class. The probability is a confidence estimate,
            not a guarantee that the prediction is correct.
        """
        self._check_trained()
        features = self.vectorizer.transform([text])
        label = int(self.model.predict(features)[0])
        confidence = float(self.model.predict_proba(features)[0, label])

        return label, confidence

    def predict_batch(self, texts: Iterable[str]) -> tuple[list[int], list[float]]:
        """Classify multiple documents in one vectorization operation.

        Returns:
            Two lists: predicted labels and the corresponding predicted-class
            probabilities, in the same order as ``texts``.
        """
        self._check_trained()
        features = self.vectorizer.transform(list(texts))
        probabilities = self.model.predict_proba(features)
        labels = self.model.predict(features).astype(int)
        confidences = probabilities[np.arange(len(labels)), labels]

        return labels.tolist(), confidences.tolist()

    def evaluate(self, texts: Iterable[str], labels: Iterable[int]) -> dict[str, float]:
        """Evaluate the already-fitted detector on labeled documents.

        This method only transforms the supplied documents with the existing
        TF-IDF vocabulary. It never refits the vectorizer or classifier, so it
        can be used for a held-out language or a different dataset.
        """
        self._check_trained()
        text_list = list(texts)
        label_array = np.asarray(list(labels), dtype=int)
        if len(text_list) != len(label_array):
            raise ValueError("texts and labels must have the same length")

        features = self.vectorizer.transform(text_list)
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1]

        return _metrics(label_array, predictions, probabilities)

    def save(self, save_dir: str | Path) -> None:
        """Save the trained components to a directory.

        ``vectorizer.pkl`` stores the learned vocabulary and IDF values,
        ``model.pkl`` stores the logistic-regression classifier, and
        ``metadata.json`` stores human-readable configuration information.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        with (save_path / "vectorizer.pkl").open("wb") as file:
            pickle.dump(self.vectorizer, file)
        with (save_path / "model.pkl").open("wb") as file:
            pickle.dump(self.model, file)
        metadata = {
            "is_trained": self.is_trained,
            "max_features": self.vectorizer.max_features,
            "ngram_range": self.vectorizer.ngram_range,
            "class_weight": self.model.class_weight,
        }
        (save_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, save_dir: str | Path) -> "BaselineDetector":
        """Reconstruct a detector saved with `save`."""
        save_path = Path(save_dir)
        with (save_path / "vectorizer.pkl").open("rb") as file:
            vectorizer = pickle.load(file)
        with (save_path / "model.pkl").open("rb") as file:
            model = pickle.load(file)
        detector = cls(
            max_features=vectorizer.max_features,
            ngram_range=vectorizer.ngram_range,
            class_weight=model.class_weight,
        )
        detector.vectorizer = vectorizer
        detector.model = model
        detector.is_trained = True
        return detector

    def _check_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError("fit the detector before making predictions")


def _metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    """Calculate the metrics used to compare detector experiments.
    """
    result = {
        "support": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }
    result["auc"] = (
        float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) == 2
        else float("nan")
    )

    return result


def _group_metrics(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    column: str,
) -> dict[str, dict[str, float | int]]:
    """Calculate metrics separately for every value in a metadata column."""
    results = {}
    for value, group in frame.groupby(column, dropna=False):
        # Convert DataFrame index labels to row positions in the prediction arrays.
        indices = frame.index.get_indexer(group.index)
        results[str(value)] = _metrics(
            group["label"].to_numpy(),
            predictions[indices],
            probabilities[indices],
        )
    return results


def _model_group_metrics(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Calculate meaningful metrics for each generator-model group.

    A generator-model group normally contains only one true class, so
    precision and ROC-AUC are not informative. Recall, F1, accuracy, and
    support are retained for model-level comparison. Human-written samples
    are excluded because ``multi_label == "human"`` is not a generator model.
    """
    return {
        model: {
            metric: values[metric]
            for metric in ("support", "accuracy", "recall", "f1")
        }
        for model, values in _group_metrics(
            frame,
            predictions,
            probabilities,
            "multi_label",
        ).items()
        if model.lower() != "human"
    }


def _frame(dataset: Any) -> Any:
    """Extract the columns needed by the detector from a project dataset.

    Both dataset classes expose their filtered pandas DataFrame as ``.df``.
    Keep all columns here because error analysis needs metadata such as the
    source, originating model, language, and text length.
    """
    return dataset.df.dropna(subset=["text", "label"]).copy()


def _save_errors(
    frame: pd.DataFrame,
    predictions: Iterable[int],
    confidences: Iterable[float],
    output_path: str | Path,
) -> int:
    """Save incorrect predictions and their original dataset metadata.

    The output contains every original column plus ``predicted_label``,
    ``confidence``, and ``correct``. CSV is used so the file can be opened
    directly in a spreadsheet or loaded again with pandas.
    """
    analysis_frame = frame.reset_index(drop=True).copy()
    analysis_frame["predicted_label"] = list(predictions)
    analysis_frame["confidence"] = list(confidences)
    analysis_frame["correct"] = analysis_frame["label"] == analysis_frame["predicted_label"]
    errors = analysis_frame.loc[~analysis_frame["correct"]].copy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(output_path, index=False)
    return len(errors)


def _sample_balanced(
    frame: pd.DataFrame,
    sample_size: int | None,
    balance_classes: bool,
    balance_languages: bool,
    balance_models: bool,
) -> pd.DataFrame:
    """Sample an equal number of rows from each selected balancing stratum.

    The selected columns define the strata. For example, enabling
    ``balance_classes`` and ``balance_languages`` groups by
    ``("label", "language")`` and therefore balances human/machine samples
    separately within each language. If no balancing options are enabled, the
    input is returned unchanged unless ``sample_size`` is set.

    ``multi_label`` and ``label`` are not independent in these datasets:
    ``multi_label == "human"`` always represents human text. Consequently,
    enabling both model and class balancing can produce very small strata or
    fail when a requested combination does not exist.
    """
    if sample_size is not None and sample_size < 2:
        raise ValueError("sample_size must be at least 2")

    balance_columns = []
    if balance_classes:
        balance_columns.append("label")
    if balance_languages:
        balance_columns.append("language")
    if balance_models:
        balance_columns.append("multi_label")

    if not balance_columns:
        if sample_size is None or sample_size >= len(frame):
            return frame.copy()
        return frame.sample(n=sample_size, random_state=42).reset_index(drop=True)

    group_sizes = frame.groupby(balance_columns, dropna=False).size()
    samples_per_group = int(group_sizes.min())
    if sample_size is not None:
        samples_per_group = min(
            samples_per_group,
            sample_size // len(group_sizes),
        )
    if samples_per_group == 0:
        raise ValueError("sample_size is too small for the selected balancing dimensions")

    parts = [
        group.sample(n=samples_per_group, random_state=42)
        for _, group in frame.groupby(balance_columns, dropna=False)
    ]
    return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)


def train_detector(
    dataset_name: str,
    csv_path: str | Path | None = None,
    languages: Iterable[str] | None = None,
    sample_size: int | None = None,
    balance_classes: bool = True,
    balance_languages: bool = False,
    balance_models: bool = False,
    exclude_noise: bool = False,
    max_features: int = 5_000,
    ngram_range: tuple[int, int] = (1, 2),
    save_dir: str | Path | None = None,
) -> BaselineDetector:
    """Train the baseline on the official training split.

    Args:
        dataset_name: Either ``"MultiSocial"`` or ``"MULTITuDE"``.
        csv_path: Optional path overriding the loader's default CSV path.
        languages: Optional language codes used to filter the training split.
        sample_size: Optional total number of training examples. With class
            balancing enabled, the budget is distributed evenly across the
            selected balancing strata.
        balance_classes: Balance human and machine labels.
        balance_languages: Balance the selected languages.
        balance_models: Balance the values in the ``multi_label`` column.
        exclude_noise: For MultiSocial, discard rows marked as potential noise.
        max_features: Maximum TF-IDF vocabulary size.
        ngram_range: Minimum and maximum n-gram sizes for TF-IDF.
        save_dir: Optional directory where the fitted detector is saved.

    Returns:
        The fitted detector.

    No test data is loaded or evaluated here. Use :func:`evaluate_detector`
    separately, which allows its language filter to differ from the training
    language filter.
    """
    key = dataset_name.lower()
    if key not in DATASET_CLASSES:
        raise ValueError(f"dataset_name must be one of: {', '.join(DATASET_CLASSES)}")

    dataset_class = DATASET_CLASSES[key]
    common_args: dict[str, Any] = {
        "languages": list(languages) if languages is not None else None,
    }
    if csv_path is not None:
        common_args["csv_path"] = csv_path
    if key == "multisocial":
        common_args["exclude_noise"] = exclude_noise

    # Load only the official training split so evaluation data cannot affect fit.
    train_frame = _frame(dataset_class(split="train", **common_args))
    train_frame = _sample_balanced(
        train_frame,
        sample_size,
        balance_classes=balance_classes,
        balance_languages=balance_languages,
        balance_models=balance_models,
    )

    detector = BaselineDetector(max_features=max_features, ngram_range=ngram_range)
    detector.fit(train_frame["text"], train_frame["label"])
    print(f"{dataset_name}: trained on {len(train_frame)} samples")

    if save_dir is not None:
        detector.save(save_dir)

    return detector


def evaluate_detector(
    detector: BaselineDetector,
    dataset_name: str,
    csv_path: str | Path | None = None,
    languages: Iterable[str] | None = None,
    sample_size: int | None = None,
    balance_classes: bool = True,
    balance_languages: bool = False,
    balance_models: bool = False,
    exclude_noise: bool = False,
    split: str = "test",
    error_analysis_path: str | Path | None = None,
    metrics_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a fitted detector on a selected dataset split.

    ``languages`` is independent from the language list passed to
    :func:`train_detector`. For example, train with ``languages=["en"]`` and
    call this function with ``languages=["nl"]`` to measure cross-language
    generalization. The detector's TF-IDF vocabulary is reused unchanged.

    Args:
        detector: A fitted :class:`BaselineDetector`.
        dataset_name: Either ``"MultiSocial"`` or ``"MULTITuDE"``.
        csv_path: Optional path overriding the loader's default CSV path.
        languages: Optional language codes used only for evaluation.
        sample_size: Optional total number of evaluation examples.
        balance_classes: Balance human and machine labels independently.
        balance_languages: Balance the selected evaluation languages.
        balance_models: Balance the values in the ``multi_label`` column.
        exclude_noise: For MultiSocial, discard rows marked as potential noise.
        split: Dataset split to evaluate, normally ``"test"``.
        error_analysis_path: Optional CSV path for saving only incorrect
            predictions together with their dataset metadata.
        metrics_output_path: Optional JSON path for saving overall,
            per-language, and per-model metrics.

    Returns:
        A dictionary containing overall metrics and separate metrics grouped
        by language. Each group also includes ``support``, the number of
        evaluated samples in that group.
    """
    key = dataset_name.lower()
    if key not in DATASET_CLASSES:
        raise ValueError(f"dataset_name must be one of: {', '.join(DATASET_CLASSES)}")

    dataset_class = DATASET_CLASSES[key]
    language_list = list(languages) if languages is not None else None
    dataset_args: dict[str, Any] = {
        "split": split,
        "languages": language_list,
    }
    if csv_path is not None:
        dataset_args["csv_path"] = csv_path
    if key == "multisocial":
        dataset_args["exclude_noise"] = exclude_noise

    frame = _frame(dataset_class(**dataset_args))
    frame = _sample_balanced(
        frame,
        sample_size,
        balance_classes=balance_classes,
        balance_languages=balance_languages,
        balance_models=balance_models,
    )

    detector._check_trained()
    features = detector.vectorizer.transform(frame["text"])
    predictions = detector.model.predict(features)
    probabilities = detector.model.predict_proba(features)[:, 1]
    confidences = probabilities * predictions + (1 - probabilities) * (1 - predictions)
    metrics = {
        "overall": _metrics(frame["label"].to_numpy(), predictions, probabilities),
        "by_language": _group_metrics(frame, predictions, probabilities, "language"),
        "by_model": _model_group_metrics(frame, predictions, probabilities),
    }
    if error_analysis_path is not None:
        error_count = _save_errors(
            frame,
            predictions,
            confidences,
            error_analysis_path,
        )
        print(f"Saved {error_count} incorrect samples to {error_analysis_path}")
    if metrics_output_path is not None:
        metrics_path = Path(metrics_output_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=True))
        print(f"Saved evaluation metrics to {metrics_path}")
    print(f"{dataset_name} ({split}, {language_list or 'all languages'}): {len(frame)} samples")
    print(json.dumps(metrics, indent=2))
    print(classification_report(frame["label"], predictions, target_names=["Human", "Machine"], zero_division=0))

    return metrics
