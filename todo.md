# TODO

Everyone chooses some detector config to find scores for all the data. Daimy the baseline detectors, the others the fastdetect/detectgpt using huggingface models. Check bottom section for Snellius tutorial.

## Daimy

TF-IDF baseline, four training data language configs. 

```python
from detector.baseline.baseline_detector import train_detector
train_detector('multisocial', save_dir='detector/baseline/models/multisocial_all')
train_detector('multisocial', languages=['en'], save_dir='detector/baseline/models/multisocial_en')
train_detector('multisocial', languages=['fr', 'de', 'en', 'es', 'pt'], save_dir='detector/baseline/models/multisocial_big_eu')
train_detector('multisocial', languages=['nl', 'et', 'ga', 'gd'], save_dir='detector/baseline/models/multisocial_less_common')
```

Write job files for all these commands and run on Snellius. 

```bash
python score.py --detector baseline --dataset multisocial --baseline-weights detector/baseline/models/multisocial_all
python score.py --detector baseline --dataset multitude --baseline-weights detector/baseline/models/multisocial_all
python score.py --detector baseline --dataset multisocial --baseline-weights detector/baseline/models/multisocial_en
python score.py --detector baseline --dataset multitude --baseline-weights detector/baseline/models/multisocial_en
python score.py --detector baseline --dataset multisocial --baseline-weights detector/baseline/models/multisocial_big_eu
python score.py --detector baseline --dataset multitude --baseline-weights detector/baseline/models/multisocial_big_eu
python score.py --detector baseline --dataset multisocial --baseline-weights detector/baseline/models/multisocial_less_common
python score.py --detector baseline --dataset multitude --baseline-weights detector/baseline/models/multisocial_less_common
```

## Ozan

Qwen/Qwen3-0.6B, fastdetect (same model as both reference and scoring) and detectgpt. 

Write job files for all these commands and run on Snellius

```bash
python score.py --detector fastdetect --dataset multisocial --scoring Qwen/Qwen3-0.6B
python score.py --detector fastdetect --dataset multitude --scoring Qwen/Qwen3-0.6B
python score.py --detector detectgpt --dataset multisocial --scoring Qwen/Qwen3-0.6B --detectgpt-mask google/mt5-small
python score.py --detector detectgpt --dataset multitude --scoring Qwen/Qwen3-0.6B --detectgpt-mask google/mt5-small
```

## Stef

Same as Ozan, one size up: Qwen/Qwen3-1.7B.

Write job files for all these commands and run on Snellius

```bash
python score.py --detector fastdetect --dataset multisocial --scoring Qwen/Qwen3-1.7B
python score.py --detector fastdetect --dataset multitude --scoring Qwen/Qwen3-1.7B
python score.py --detector detectgpt --dataset multisocial --scoring Qwen/Qwen3-1.7B --detectgpt-mask google/mt5-small
python score.py --detector detectgpt --dataset multitude --scoring Qwen/Qwen3-1.7B --detectgpt-mask google/mt5-small
```

## Julian

gpt2-xl (1.5B), monolingual English model, bigger like Qwen-1.5B.

Write job files for all these commands and run on Snellius

```bash
python score.py --detector fastdetect --dataset multisocial --scoring gpt2-xl
python score.py --detector fastdetect --dataset multitude --scoring gpt2-xl
python score.py --detector detectgpt --dataset multisocial --scoring gpt2-xl --detectgpt-mask google/mt5-small
python score.py --detector detectgpt --dataset multitude --scoring gpt2-xl --detectgpt-mask google/mt5-small
```

## Pepijn

gpt2 (124M), monolingual English model, small like Qwen-0.5B.

Write job files for all these commands and run on Snellius

```bash
python score.py --detector fastdetect --dataset multisocial --scoring gpt2
python score.py --detector fastdetect --dataset multitude --scoring gpt2
python score.py --detector detectgpt --dataset multisocial --scoring gpt2 --detectgpt-mask google/mt5-small
python score.py --detector detectgpt --dataset multitude --scoring gpt2 --detectgpt-mask google/mt5-small
```

# Running on Snellius

[https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial1/Lisa_Cluster.html](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial1/Lisa_Cluster.html)

```bash
ssh <username>@snellius.surf.nl
git clone https://github.com/stefdewildt/Multi-lingual-NLP.git
cd Multi-lingual-NLP

# datasets aren't in git, copy them from wherever you already have them
rsync -avz datasets/MultiSocial/multisocial.csv <username>@snellius.surf.nl:~/Multi-lingual-NLP/datasets/MultiSocial/
rsync -avz datasets/MULTITuDE/multitude.csv <username>@snellius.surf.nl:~/Multi-lingual-NLP/datasets/MULTITuDE/
```

```bash
module purge
module load 2025
module load Python/3.13.5-GCCcore-14.3.0
cd $HOME/Multi-lingual-NLP
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# compute nodes have no internet: pre-download every model from the login node before queueing any job
export HF_HOME=$HOME/hf_cache
python -c "
from detector.models import load_model, load_tokenizer
load_model('Qwen/Qwen3-0.6B', 'causal', 'cpu')
load_tokenizer('Qwen/Qwen3-0.6B')
load_model('google/mt5-small', 'seq2seq', 'cpu')
load_tokenizer('google/mt5-small')
"
```

```bash
#!/bin/bash
#SBATCH --job-name=fastdetect_qwen0.5b
# partitions: https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius+partitions+and+accounting
#SBATCH --partition=gpu_a100   # baseline (Daimy): rome instead, it never uses a GPU
#SBATCH --gpus=1               # baseline (Daimy): remove this line
#SBATCH --cpus-per-task=18     # gpu_a100 share (9 for gpu_mig, 16 for gpu_h100). baseline on rome: 16
#SBATCH --mem=120G             # gpu_a100 share (60G for gpu_mig, 180G for gpu_h100), fixed per GPU regardless of model. baseline on rome: 28G
#SBATCH --time=08:00:00        # unverified guess, but note that detectgpt is slower than fastdetect and you can always resume the script if time runs out. time a --dataset-limit run first if unsure.
#SBATCH --output=slurm-%j.out

module purge
module load 2025
module load Python/3.13.5-GCCcore-14.3.0
cd $HOME/Multi-lingual-NLP
source .venv/bin/activate
export HF_HOME=$HOME/hf_cache
export HF_HUB_OFFLINE=1

# make and queue separate job files for different runs
# run once with --dataset-limit 20 first as if you wanna test first
srun python score.py --detector fastdetect --dataset multisocial --scoring Qwen/Qwen3-0.6B
```

```bash
sbatch job_fastdetect_qwen0.5b.sh
squeue -u $USER
```
