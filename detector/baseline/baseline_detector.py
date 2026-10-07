"""TF-IDF and logistic-regression baseline for AI-generated text detection.

The baseline represents each document as a sparse vector of word n-gram
TF-IDF scores and uses logistic regression to predict a binary label:

0 means human-written text.
1 means machine-generated text.

The `train_detector` function can be used to train the model with the project's
MultiSocial and MULTITuDE dataset loaders. 
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

from datasets.multisocial import MultiSocialDataset
from datasets.multitude import MultitudeDataset

# easy to write/read dataset names
DATASET_CLASSES = {
    "multisocial": MultiSocialDataset,
    "multitude": MultitudeDataset,
}

# These names let one detector learn from both datasets at once.
COMBINED_DATASET_NAMES = {"both", "combined"}


class BaselineDetector:
    """Binary TF-IDF classifier where 0 is human and 1 is machine-generated.

    The vectorizer learns the vocabulary and inverse-document frequencies from
    training text. The logistic-regression model then learns how useful those
    features are for separating the two classes.
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
        # initialize the tf-idf vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            lowercase=True,
        )

        # initialize the logistic regression classifier
        self.model = LogisticRegression(
            class_weight=class_weight,
            max_iter=1_000,
            random_state=42,
        )

        # store the fact that the current detector has not yet been trained, so predictions are not allowed
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
            This detector
        """
        text_list = list(texts)
        label_array = np.asarray(list(labels), dtype=int)

        # each text needs exactly one matching label
        if len(text_list) != len(label_array):
            raise ValueError("texts and labels must have the same length")

        # make sure we have at least one example of each class, otherwise the classifier will fail
        if len(np.unique(label_array)) < 2:
            raise ValueError("training data must contain both classes")

        # learn the vocabulary (features) first, then train the classifier on those features
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

        # get the TF-IDF features for the input text, then predict the label
        features = self.vectorizer.transform([text])
        label = int(self.model.predict(features)[0])

        # find the probability of belonging to the selected label (this shows how confident the model is in its prediction)
        confidence = float(self.model.predict_proba(features)[0, label])

        return label, confidence

    def save(self, save_dir: str | Path) -> None:
        """Save the trained components to a directory.

        vectorizer.pkl stores the learned vocabulary and IDF values,
        model.pkl stores the logistic-regression classifier, and
        metadata.json stores human-readable configuration information.
        """
        # create the destination folder when it does not exist
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # save the vocabulary and the trained classifier separately
        with (save_path / "vectorizer.pkl").open("wb") as file:
            pickle.dump(self.vectorizer, file)
        with (save_path / "model.pkl").open("wb") as file:
            pickle.dump(self.model, file)

        # store the main settings so the saved files are easier to understand
        metadata = {
            "is_trained": self.is_trained,
            "max_features": self.vectorizer.max_features,
            "ngram_range": self.vectorizer.ngram_range,
            "class_weight": self.model.class_weight,
        }
        (save_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, save_dir: str | Path) -> "BaselineDetector":
        """Load a detector that was previously saved."""
        # read both trained parts from the folder created by the save function. 
        save_path = Path(save_dir)

        with (save_path / "vectorizer.pkl").open("rb") as file:
            vectorizer = pickle.load(file)

        with (save_path / "model.pkl").open("rb") as file:
            model = pickle.load(file)

        # build a detector with matching settings
        detector = cls(
            max_features=vectorizer.max_features,
            ngram_range=vectorizer.ngram_range,
            class_weight=model.class_weight,
        )

        # make sure the restored detector is ready for predictions
        detector.vectorizer = vectorizer
        detector.model = model
        detector.is_trained = True

        return detector

    def _check_trained(self) -> None:
        """Raise a clear error when prediction is requested but training has not been done."""
        if not self.is_trained:
            raise RuntimeError("fit the detector before making predictions")



def _frame(dataset: Any) -> Any:
    """Extract the columns needed by the detector from a dataset.
    """
    # remove incomplete rows so training and evaluation receive usable text
    return dataset.df.dropna(subset=["text", "label"]).copy()


def _sample_balanced(
    frame: pd.DataFrame,
    sample_size: int | None,
    balance_classes: bool,
    balance_languages: bool,
    balance_models: bool,
) -> pd.DataFrame:
    """Sample an equal number of rows from each selected balancing stratum.
    Note: we coded this possibility, but decided not to use it in the experiments.
    As this model only serves as a baseline, spending time tuning hyperparameters and finding
    which balancing strategy works best was not a priority.
    """
    # a small sample would not leave enough room for both classes
    if sample_size is not None and sample_size < 2:
        raise ValueError("sample_size must be at least 2")

    # collect the columns that should have equally sized groups
    balance_columns = []
    if balance_classes:
        balance_columns.append("label")
    if balance_languages:
        balance_columns.append("language")
    if balance_models:
        balance_columns.append("multi_label")

    # without balancing, either keep all rows or take a repeatable random sample
    if not balance_columns:
        if sample_size is None or sample_size >= len(frame):
            return frame.copy()
        return frame.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # find the smallest group so every group can contribute the same amount
    group_sizes = frame.groupby(balance_columns, dropna=False).size()
    samples_per_group = int(group_sizes.min())

    # spread a requested sample size across all selected groups
    if sample_size is not None:
        samples_per_group = min(
            samples_per_group,
            sample_size // len(group_sizes),
        )
    if samples_per_group == 0:
        raise ValueError("sample_size is too small for the selected balancing dimensions")

    # take the same number from each group, then mix the groups together.
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
    balance_classes: bool = False,
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

    Notes:
        Use ``dataset_name="both"`` or ``dataset_name="combined"`` to train
        on the training split from both datasets. In combined mode, leave
        ``csv_path`` as ``None`` so each dataset can use its own default file.
    """
    # normalize the dataset name so users can write it with any capitalization
    key = dataset_name.lower()
    is_combined = key in COMBINED_DATASET_NAMES
    if not is_combined and key not in DATASET_CLASSES:
        valid_names = [*DATASET_CLASSES, *sorted(COMBINED_DATASET_NAMES)]
        raise ValueError(f"dataset_name must be one of: {', '.join(valid_names)}")

    # a single custom path cannot describe two different dataset files
    if is_combined and csv_path is not None:
        raise ValueError(
            "csv_path must be None when training on both datasets; "
            "each dataset uses its own default CSV path"
        )

    # build options shared by one dataset or by both datasets.
    common_args: dict[str, Any] = {
        "languages": list(languages) if languages is not None else None,
    }

    # keep the dataset-specific noise option out of MULTITuDE.
    if not is_combined and key == "multisocial":
        common_args["exclude_noise"] = exclude_noise

    # choose one loader or both loaders, depending on the requested mode.
    dataset_keys = ["multisocial", "multitude"] if is_combined else [key]
    train_frames = []

    # load the requested datasets and extract the columns needed by the detector
    for dataset_key in dataset_keys:
        dataset_args = common_args.copy()
        if csv_path is not None:
            dataset_args["csv_path"] = csv_path
        if dataset_key == "multisocial":
            dataset_args["exclude_noise"] = exclude_noise

        # use training data only, so evaluation data cannot influence the fit.
        dataset_class = DATASET_CLASSES[dataset_key]
        train_frames.append(_frame(dataset_class(split="train", **dataset_args)))

    # put both datasets into one table with the same row structure.
    train_frame = pd.concat(train_frames, ignore_index=True)

    # apply the requested sample size and balancing choices.
    train_frame = _sample_balanced(
        train_frame,
        sample_size,
        balance_classes=balance_classes,
        balance_languages=balance_languages,
        balance_models=balance_models,
    )

    # train the detector on the prepared text and labels
    detector = BaselineDetector(max_features=max_features, ngram_range=ngram_range)
    detector.fit(train_frame["text"], train_frame["label"])

    # tell the user how many examples were used for this run
    print(f"{dataset_name}: trained on {len(train_frame)} samples")

    # saving is optional, so normal experiments can keep the model in memory
    if save_dir is not None:
        detector.save(save_dir)

    return detector