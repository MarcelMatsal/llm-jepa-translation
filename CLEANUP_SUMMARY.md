# Code Cleanup Summary

## Removed Files

✅ **Deleted `src/utils/` directory**
- `src/utils/__init__.py` - Not used anywhere
- `src/utils/config.py` - Not used anywhere

## Simplified Code

### `src/data/datasets.py`
**Removed:**
- `process_custom_jsonl()` function - JSONL support removed
- `processor_fn` parameter - Custom processors removed
- Complex format handling for multiple dataset types
- `Optional` import (no longer needed)

**Simplified:**
- `get_dataset()` now only takes `lang_pair` (no `dataset_name` parameter)
- Only handles WMT19 format: `{"translation": {"lang1": "...", "lang2": "..."}}`
- Cleaner, focused code (93 lines vs 201 lines)

### `scripts/train.py`
**Removed:**
- `cfg.data.dataset` validation (no longer needed)
- Multiple dataset format fallbacks

**Simplified:**
- Hardcoded to use `"wmt19"` dataset
- Only requires `lang_pair` in config

### `scripts/evaluate.py`
**Removed:**
- `--dataset` argument (always WMT19)
- Custom dataset loading logic

**Simplified:**
- Only requires `--lang_pair` argument
- Uses WMT19 automatically

### `configs/default.yaml`
**Removed:**
- `dataset` field (always WMT19)

**Simplified:**
- Only requires `lang_pair` in data section

## Current Structure

```
src/
├── models/          # Model architectures
├── data/            # WMT19 dataset loader only
├── training/         # Training utilities
└── (no utils/)      # ✅ Removed

scripts/
├── train.py         # Simplified, config-only
└── evaluate.py      # Simplified, WMT19 only

configs/
└── default.yaml     # Simplified config
```

## Usage

```bash
# Train
python scripts/train.py --config configs/default.yaml

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/checkpoint.pt --lang_pair en-de
```

## Next Steps

To add more datasets later:
1. Add new dataset-specific loader functions
2. Add dataset selection to config
3. Update `get_dataset()` to route to appropriate loader

