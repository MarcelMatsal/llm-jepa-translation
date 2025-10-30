# Multilingual JEPA Implementation Plan

## Architecture Overview

### Core Components
1. **X-Encoder & Y-Encoder**: Separate transformer encoders (one per language)
2. **Predictor**: Single shared predictor conditioned on language pair
3. **Training**: Bidirectional (both directions), EMA updates for y-encoder
4. **Loss**: MSE on normalized embeddings

## Project Structure

```
llm-jepa-translation/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py          # Transformer encoder with CLS token
│   │   ├── predictor.py        # Language-conditioned predictor
│   │   └── jepa.py             # Main JEPA model (x/y encoders + predictor)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py         # Generic dataset loaders
│   │   ├── collators.py        # Data collation for batching
│   │   └── processors.py       # Dataset format processors
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop with EMA
│   │   └── metrics.py          # Evaluation metrics
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Configuration management
│       └── logging.py           # Experiment tracking
├── configs/
│   └── default.yaml            # Default configuration
├── scripts/
│   ├── train.py                # Main training script
│   └── evaluate.py             # Evaluation script
├── experiments/
│   └── README.md               # Experiment tracking
├── requirements.txt
└── README.md
```

## Implementation Details

### 1. Models (`src/models/`)

#### `encoder.py`
- Transformer encoder (BERT-style or configurable)
- CLS token extraction
- Pooling fallback (mean/attention) if no CLS token

#### `predictor.py`
- MLP with language conditioning
- Input: `s_x` (d_model) + `z` (lang_emb_src + lang_emb_tgt)
- Output: `s_y_pred` (d_model)

#### `jepa.py`
- Main model combining x-encoder, y-encoder, predictor
- EMA update logic
- Forward pass for both directions

### 2. Data (`src/data/`)

#### `datasets.py`
- Generic HuggingFace dataset loader
- Supports multiple formats:
  - Translation pairs (e.g., WMT19, OPUS)
  - Custom JSONL with `text_src`, `text_tgt`, `lang_src`, `lang_tgt`
  - Any format via processor functions

#### `processors.py`
- Format-specific processors:
  - `TranslationProcessor`: For translation datasets
  - `CustomProcessor`: For JSONL files
  - Easy to extend for new formats

#### `collators.py`
- Batch collation with padding
- Handles variable-length sequences
- Returns batched tensors + language IDs

### 3. Training (`src/training/`)

#### `trainer.py`
- Training loop (can integrate with stable-pretraining)
- Bidirectional loss computation
- EMA updates after each step
- Gradient clipping, logging

#### `metrics.py`
- Embedding similarity (cosine, L2)
- Linearity checks (SVD of difference)
- Embedding diversity metrics

### 4. Configuration (`configs/`)

YAML-based configs for:
- Model architecture
- Training hyperparameters
- Dataset paths/formats
- Experiment tracking

## Key Design Decisions

1. **CLS Token Priority**: Use CLS token if available, fallback to pooling
2. **One Predictor**: Single predictor conditioned on language pair IDs
3. **Bidirectional**: Train both directions simultaneously
4. **EMA**: Update y-encoder weights via EMA after each step
5. **Normalization**: Always normalize embeddings before loss
6. **Flexible Datasets**: Processor pattern for different formats

## Dependencies

- torch
- transformers
- datasets
- stable-pretraining (for training management)
- omegaconf (for configs)
- tqdm (for progress bars)

