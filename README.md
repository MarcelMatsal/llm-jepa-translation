# Multilingual JEPA: Learning Aligned Embeddings Across Languages

Joint Embedding Predictive Architecture for multilingual sentence embeddings.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train (uses config.yaml in root, or specify another)
python scripts/train.py --config config.yaml
```

## Architecture

- **Separate Encoders**: One encoder per language (x-encoder for source, y-encoder for target)
- **EMA Updates**: y-encoder updated via Exponential Moving Average of x-encoder
- **Predictor**: Single language-conditioned predictor for all language pairs
- **Bidirectional**: Trains both directions (lang1→lang2 and lang2→lang1)

## Project Structure

- `src/models/`: Model architectures (encoders, predictor, JEPA)
- `src/data/`: WMT19 dataset loader
- `src/training/`: Training loop and metrics
- `experiments/`: Experimental results and outputs
- `scripts/`: Training and evaluation scripts
- `config.yaml`: Default config template

Configs are saved with each experiment automatically.
