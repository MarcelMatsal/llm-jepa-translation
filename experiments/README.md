# Experiments

Storage directory for experimental results.

## Structure

```
experiments/
└── exp_test/
    ├── checkpoint.pt      # Model weights
    ├── config.yaml        # Config used (auto-saved)
    └── notes.txt          # Your notes (optional)
```

## Usage

### Run experiment

```bash
python scripts/train.py --config experiments/exp_test/config.yaml
```

### Create new experiment

```bash
# Copy config from previous experiment
cp experiments/exp_test/config.yaml exp_002_config.yaml

# Edit exp_002_config.yaml (change lang_pair, save_dir, etc.)
python scripts/train.py --config exp_002_config.yaml
```

### Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint experiments/exp_test/checkpoint.pt \
    --lang_pair lt-en
```

## What Gets Saved

- `checkpoint.pt` - Model weights (gitignored)
- `config.yaml` - Config used (auto-saved, can commit)
