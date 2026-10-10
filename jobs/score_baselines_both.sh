#!/bin/bash
# train the combined-dataset baselines and score them on both test splits
set -e
python -u -m detector.baseline.train_both
for set in all en big_eu less_common; do
  for dataset in multisocial multitude; do
    python -u score.py --detector baseline --dataset $dataset --dataset-split test \
      --baseline-weights detector/baseline/models/both_$set
  done
done
