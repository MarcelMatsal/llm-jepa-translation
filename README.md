# Multilingual JEPA: Learning Aligned Embeddings Across Languages

Joint Embedding Predictive Architecture for multilingual sentence embeddings.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train on translation dataset
python scripts/train.py --config configs/default.yaml
```

## Architecture

- **Separate Encoders**: One encoder per language (x-encoder for source, y-encoder for target)
- **EMA Updates**: y-encoder updated via Exponential Moving Average of x-encoder
- **Predictor**: Single language-conditioned predictor for all language pairs
- **Bidirectional**: Trains both directions (lang1→lang2 and lang2→lang1)

## Project Structure

- `src/models/`: Model architectures (encoders, predictor, JEPA)
- `src/data/`: Dataset loaders and processors
- `src/training/`: Training loop and metrics
- `configs/`: Configuration files
- `scripts/`: Training and evaluation scripts

See `IMPLEMENTATION_PLAN.md` for details.
