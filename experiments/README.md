# Experiments

Storage directory for experimental results and outputs.

## What Goes Here

After training, save your experiment results here:

```
experiments/
├── exp_001_wmt19_en_de/
│   ├── checkpoint.pt      # Model weights (gitignored)
│   ├── config.yaml        # Config used (auto-saved, can commit)
│   └── notes.txt          # Your notes about results (optional)
├── exp_002_wmt19_cs_en/
│   └── ...
└── README.md
```

## How to Use

**Option 1: Save directly to experiments**
```yaml
# Create a config file (e.g., exp_001_config.yaml)
output:
  save_dir: "./experiments/exp_001_wmt19_en_de"
```
```bash
python scripts/train.py --config exp_001_config.yaml
```

**Option 2: Use default config, move after training**
```bash
python scripts/train.py --config config.yaml
mv checkpoints experiments/exp_001_wmt19_en_de
```

**Option 3: Copy config from previous experiment**
```bash
cp experiments/exp_001/config.yaml exp_002_config.yaml
# Edit exp_002_config.yaml (change lang_pair, save_dir, etc.)
python scripts/train.py --config exp_002_config.yaml
```

## What Gets Saved Automatically

When you train, these are saved automatically:
- `checkpoint.pt` - Model weights and optimizer state (gitignored)
- `config.yaml` - The config file used for training (can commit)

## Optional: Save Metrics

After evaluation, save metrics manually:
```bash
python scripts/evaluate.py \
    --checkpoint experiments/exp_001/checkpoint.pt \
    --lang_pair en-de \
    > experiments/exp_001/metrics.txt
```

## Naming Experiments

Use descriptive names:
- `exp_001_wmt19_en_de_baseline`
- `exp_002_wmt19_cs_en_attention_pool`
- `exp_003_wmt19_fr_de_tau_0999`

**Note**: Checkpoints (`.pt`, `.pth`) are gitignored, but you can commit `config.yaml` files and notes to track your experiments.
