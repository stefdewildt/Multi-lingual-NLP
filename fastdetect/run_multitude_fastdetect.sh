#!/usr/bin/env bash

set -euo pipefail

FASTDETECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$FASTDETECT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"

# use the same model for both roles in the experiment
REFERENCE_MODELS=("sshleifer/tiny-gpt2")
SCORING_MODELS=("sshleifer/tiny-gpt2")

# add more languages here
LANGUAGES=("en" "nl")

CSV_PATH="$REPO_ROOT/datasets/MULTITuDE/multitude.csv"
CACHE_DIR="$REPO_ROOT/cache"
OUTPUT_DIR="$FASTDETECT_DIR/multitude_experiments/results"
SPLIT="test"
DEVICE="cuda"

# max human and machine rows per language, so 100 human + 100 AI samples
# use 0 for all rows
MAX_SAMPLES_PER_CLASS=100

for REFERENCE_MODEL in "${REFERENCE_MODELS[@]}"; do
    for SCORING_MODEL in "${SCORING_MODELS[@]}"; do
        for LANGUAGE in "${LANGUAGES[@]}"; do
            # replace the huggingface "/" with "_" to support the output file naming
            reference_name="${REFERENCE_MODEL//\//_}"
            scoring_name="${SCORING_MODEL//\//_}"
            output_file="$OUTPUT_DIR/${reference_name}_${scoring_name}_${LANGUAGE}.csv"

            # run the evaluation on this config
            "$PYTHON" "$FASTDETECT_DIR/scripts/multitude_fast_detect.py" \
                --csv_path "$CSV_PATH" \
                --output_file "$output_file" \
                --split "$SPLIT" \
                --languages "$LANGUAGE" \
                --max_samples_per_class "$MAX_SAMPLES_PER_CLASS" \
                --sampling_model_name "$REFERENCE_MODEL" \
                --scoring_model_name "$SCORING_MODEL" \
                --device "$DEVICE" \
                --cache_dir "$CACHE_DIR"
        done
    done
done
