# Baseline Detector for Machine-Generated Text

## Overview

This module provides a simple baseline classifier for detecting machine-generated vs human-written text from the MultiSocial and MULTITuDE datasets. It uses **TF-IDF features** combined with **Logistic Regression** for fast training and inference.

## Quick Start

### Training on English Data

```python
from pathlib import Path
from detector.baseline.baseline_detector import train_detector

detector = train_detector(
    "MultiSocial",
    languages=["en"],
    sample_size=5000,
    save_dir=Path("detector/models/baseline_multisocial_en"),
)
```

### How the baseline works

1. `datasets/multisocial.py` or `datasets/multitude.py` loads and filters the CSV.
2. The training split is optionally restricted to selected languages and balanced by label, language, and/or generator model.
3. `TfidfVectorizer` converts word unigrams and bigrams into sparse features.
4. `LogisticRegression` learns label `0` = human and label `1` = machine-generated.

Run both datasets from the project directory with:

```bash
python run_baseline.py
```

The CSV files must first be downloaded according to `datasets/MultiSocial/README.md` and `datasets/MULTITuDE/README.md`.

### Making Predictions

```python
label, confidence = detector.predict("This is a test message")
print(f"Prediction: {'MACHINE' if label == 1 else 'HUMAN'} ({confidence:.2%})")
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
- `predict()`: Single prediction with confidence score
- `save()`: Save model and vectorizer to disk
- `load()`: Load pre-trained model

## Data Requirements

The detector expects:
- **Input**: Text (string)
- **Label**: Binary (0 = human-written, 1 = machine-generated)
- **Format**: Can work with raw text or pandas DataFrames

