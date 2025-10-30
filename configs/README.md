# Configuration Guide

## Usage

Train using a config file:

```bash
python scripts/train.py --config configs/default.yaml
```

## Config File Structure

All settings are in YAML format:

```yaml
# Model configuration
model:
  encoder_name: "bert-base-multilingual-cased"
  pooling: "cls"  # cls, mean, or attention
  num_languages: 9
  tau: 0.999  # EMA decay rate

# Dataset configuration (WMT19 only)
data:
  lang_pair: "en-de"  # Language pair (e.g., en-de, cs-en, fr-de)
  batch_size: 8
  num_workers: 0

# Training configuration
training:
  epochs: 1
  learning_rate: 1e-4
  max_grad_norm: 1.0
  log_interval: 100

# Output configuration
output:
  save_dir: "./checkpoints"
  device: "cuda"  # auto-detected if "cuda"
```

## Available Language Pairs in WMT19

- `en-de` (English-German)
- `cs-en` (Czech-English)
- `fi-en` (Finnish-English)
- `fr-de` (French-German)
- `ru-en` (Russian-English)
- `zh-en` (Chinese-English)
- And more - see https://huggingface.co/datasets/wmt/wmt19

## Creating Custom Configs

Copy and modify the default config:

```bash
cp configs/default.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml
python scripts/train.py --config configs/my_experiment.yaml
```

Config files are automatically saved with checkpoints for reproducibility.
