# Multilingual JEPA

Joint Embedding Predictive Architecture for learning aligned sentence embeddings across languages.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train

```bash
python scripts/train.py --config experiments/exp_test/config.yaml
```

### Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint experiments/exp_test/checkpoint.pt \
    --lang_pair lt-en
```

## Architecture

- **Separate encoders**: X-encoder (online) and Y-encoder (target, EMA-updated)
- **Predictor**: Single language-conditioned predictor for all language pairs
- **Bidirectional training**: Both directions (lang1→lang2 and lang2→lang1)
- **WMT19 dataset**: Supports all WMT19 language pairs

## Project Structure

```
src/
├── models/          # JEPA model (encoders, predictor)
├── data/            # WMT19 dataset loader
└── training/         # Training loop and metrics

experiments/         # Experimental results and configs
scripts/             # Training and evaluation scripts
```
