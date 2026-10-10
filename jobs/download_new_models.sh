#!/bin/bash
# Run on the Snellius LOGIN node (compute nodes have no internet), before
# submitting calib_new_models.job. Only downloads, doesn't load the models.
# Llama is gated: needs a Hugging Face token with approved Llama 3.2 access
# (huggingface-cli login), see archive/whichmodels.md.
set -eo pipefail
cd "$HOME/Multi-lingual-NLP"
source .venv/bin/activate
export HF_HOME="$HOME/hf_cache"

python -c "
from huggingface_hub import snapshot_download
for model in 'Qwen/Qwen3-1.7B Qwen/Qwen3-4B meta-llama/Llama-3.2-3B-Instruct utter-project/EuroLLM-1.7B HuggingFaceTB/SmolLM2-1.7B HuggingFaceTB/SmolLM2-360M'.split():
    try:
        print(model, snapshot_download(model))
    except Exception as error:  # e.g. the gated Llama without a token
        print('FAILED', model, type(error).__name__, error)
"
