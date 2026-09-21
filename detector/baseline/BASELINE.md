# Baseline Detector for Machine-Generated Text

## Overview

This module provides a simple baseline classifier for detecting machine-generated vs human-written text from the MultiSocial and MULTITuDE datasets. It uses **TF-IDF features** combined with **Logistic Regression** for fast training and inference.

## Quick Start

### Training on English Data

```python
from pathlib import Path
from detector.baseline.baseline_detector import evaluate_detector, train_detector

detector = train_detector(
    "MultiSocial",
    languages=["en"],
    sample_size=5000,
    save_dir=Path("detector/models/baseline_multisocial_en"),
)
metrics = evaluate_detector(detector, "MultiSocial", languages=["en"])
```

### Saving errors for analysis

Pass `error_analysis_path` when evaluating to save only the incorrect
predictions:

```python
metrics = evaluate_detector(
    detector,
    "MultiSocial",
    languages=["nl"],
    error_analysis_path="detector/error_analysis/multisocial_nl_errors.csv",
    metrics_output_path="detector/results/multisocial_nl_metrics.json",
)
```

`metrics_output_path` saves the complete evaluation result as JSON, including
the overall metrics, per-language metrics, and per-generator-model metrics.
The error-analysis CSV and metrics JSON are separate files so they can be
reviewed independently.

The CSV keeps the original dataset fields, including `text`, `label`,
`multi_label`, `language`, `length`, and `source`, and adds:

- `predicted_label`: the detector's prediction
- `confidence`: probability assigned to the predicted class
- `correct`: always `False` in this error-only file

This makes it possible to group errors by originating model or human text,
language, source, and text length. For example:

```python
import pandas as pd

errors = pd.read_csv("detector/error_analysis/multisocial_nl_errors.csv")
print(errors.groupby("multi_label").size().sort_values(ascending=False))
print(errors.groupby("language").size())
```

Use `"MULTITuDE"` instead of `"MultiSocial"` to run the same baseline with the MULTITuDE loader. Both dataset loaders use their official `train` and `test` split columns; TF-IDF vocabulary is fit on `train` only.

### How the baseline works

1. `datasets/multisocial.py` or `datasets/multitude.py` loads and filters the CSV.
2. The training split is optionally restricted to selected languages and balanced by label, language, and/or generator model.
3. `TfidfVectorizer` converts word unigrams and bigrams into sparse features.
4. `LogisticRegression` learns label `0` = human and label `1` = machine-generated.
5. `evaluate_detector` scores a selected split with overall metrics and metrics per language and generator model.

The returned evaluation dictionary has this structure:

```python
{
  "overall": {"accuracy": ..., "precision": ..., "recall": ..., "f1": ..., "auc": ..., "support": ...},
  "by_language": {"en": {...}, "nl": {...}},
  "by_model": {"gpt-3.5-turbo-0125": {"accuracy": ..., "recall": ..., "f1": ..., "support": ...}},
}
```

`support` is the number of evaluated samples in that group. This makes it
possible to compare, for example, whether performance differs between English
and Dutch or between generator models. The human group is excluded from
`by_model`; model-level metrics intentionally omit
precision and ROC-AUC because each model group normally contains only one true
class. Generator-model information also remains available in the saved
error-analysis CSV files.

Training and evaluation are intentionally separate. This supports cross-language experiments:

```python
from detector.baseline.baseline_detector import evaluate_detector, train_detector

detector = train_detector("MultiSocial", languages=["en"])
cross_language_metrics = evaluate_detector(
    detector,
    "MultiSocial",
    languages=["nl"],
)
```

Run both datasets from the project directory with:

```bash
python run_baseline.py
```

The CSV files must first be downloaded according to `datasets/MultiSocial/README.md` and `datasets/MULTITuDE/README.md`.

### Making Predictions

```python
# Single prediction
label, confidence = detector.predict("This is a test message")
print(f"Prediction: {'MACHINE' if label == 1 else 'HUMAN'} ({confidence:.2%})")

# Batch predictions
labels, confidences = detector.predict_batch([
    "First text",
    "Second text",
    "Third text"
])
```

### Loading a Pre-trained Model

```python
from detector.baseline.baseline_detector import BaselineDetector

detector = BaselineDetector.load(Path("detector/models/baseline_en"))
label, confidence = detector.predict("New text to classify")
```

## Architecture

### BaselineDetector Class

**Components:**
1. **TfidfVectorizer**: Converts text to numerical features
   - Max features: 5,000
   - N-grams: 1-2 (unigrams and bigrams)
   - Min document frequency: 2
   - Max document frequency: 95%

2. **LogisticRegression**: Binary classifier
   - Simple, interpretable model
   - Fast training and inference
   - Works well with TF-IDF features

**Methods:**
- `fit()`: Fit TF-IDF and logistic regression on labeled text
- `evaluate()`: Evaluate a fitted detector without refitting it
- `predict()`: Single prediction with confidence score
- `predict_batch()`: Batch predictions
- `save()`: Save model and vectorizer to disk
- `load()`: Load pre-trained model

## Data Requirements

The detector expects:
- **Input**: Text (string)
- **Label**: Binary (0 = human-written, 1 = machine-generated)
- **Format**: Can work with raw text or pandas DataFrames

