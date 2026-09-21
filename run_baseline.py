from pathlib import Path

from detector.baseline.baseline_detector import evaluate_detector, train_detector

# I can add argparse to make it easy to run from terminal, but for now parameters are just hardcoded

if __name__ == "__main__":
    # Train and evaluate the baseline detector on the MultiSocial and MULTITuDE datasets.
    # In this example, we train the detector on English data and evaluate it on both English and Dutch data.
    for dataset in ("MultiSocial", "MULTITuDE"):
        detector = train_detector(
            dataset,
            languages=["en"],
            sample_size=2_000,
            balance_classes=True,
            balance_languages=False,
            balance_models=False,
            max_features=2_000,
            ngram_range=(1, 2),
            save_dir=Path("detector/baseline/models") / f"baseline_{dataset.lower()}",
        )
        evaluate_detector(
            detector,
            dataset,
            languages=["en"],
            sample_size=2_000,
            balance_classes=True,
            balance_languages=False,
            balance_models=False,
            error_analysis_path=Path("detector/baseline/error_analysis") / f"{dataset.lower()}_en_errors.csv",
            metrics_output_path=Path("detector/baseline/results") / f"{dataset.lower()}_en_metrics.json",
        )
        evaluate_detector(
            detector,
            dataset,
            languages=["nl"],
            sample_size=2_000,
            balance_classes=True,
            balance_languages=False,
            balance_models=False,
            error_analysis_path=Path("detector/baseline/error_analysis") / f"{dataset.lower()}_nl_errors.csv",
            metrics_output_path=Path("detector/baseline/results") / f"{dataset.lower()}_nl_metrics.json",
        )
        evaluate_detector(
            detector,   
            dataset,
            languages=["en", "nl"],
            sample_size=2_000,
            balance_classes=True,
            balance_languages=True,
            balance_models=False,
            error_analysis_path=Path("detector/baseline/error_analysis") / f"{dataset.lower()}_en_nl_errors.csv",
            metrics_output_path=Path("detector/baseline/results") / f"{dataset.lower()}_en_nl_metrics.json",
        )
