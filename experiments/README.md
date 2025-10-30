# Example Usage

## Quick Start

```bash
# Train on WMT19 English-German
python scripts/train.py \
    --dataset wmt19 \
    --lang_pair en-de \
    --encoder_name bert-base-multilingual-cased \
    --epochs 10 \
    --batch_size 32

# Evaluate
python scripts/evaluate.py \
    --checkpoint ./checkpoints/checkpoint.pt \
    --dataset wmt19 \
    --lang_pair en-de
```

## Supported Datasets

- HuggingFace translation datasets:
  - `wmt19` (various language pairs)
  - `opus_books` (book translations)
  - Any dataset with language pair format

- Custom JSONL format:
```json
{"text_src": "Hello", "text_tgt": "Bonjour", "lang_src": "en", "lang_tgt": "fr"}
```

## Architecture Notes

- **CLS Token**: Uses CLS token for fixed-size embeddings (default)
- **Pooling Fallback**: Falls back to mean/attention pooling if no CLS token
- **Bidirectional**: Trains both directions simultaneously
- **EMA**: Y-encoder updated via Exponential Moving Average (tau=0.999)
- **One Predictor**: Single predictor conditioned on language pair IDs

## Key Files

- `src/models/jepa.py`: Main JEPA model
- `src/models/encoder.py`: Transformer encoder with CLS token
- `src/models/predictor.py`: Language-conditioned predictor
- `src/data/datasets.py`: Flexible dataset loaders
- `src/training/trainer.py`: Training loop with EMA
- `scripts/train.py`: Main training script

