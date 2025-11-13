# Dual-Objective BERT for Cross-Lingual Alignment

A novel approach to learning aligned cross-lingual sentence representations by training BERT-style models with dual objectives: **Masked Language Modeling (MLM)** and **CLS Token Alignment**.

Inspired by Joint Embedding Predictive Architecture (JEPA) principles, this project aims to create semantically meaningful cross-lingual embeddings by explicitly training models to align CLS token representations across languages.

## 🎯 Research Goal

Pre-trained multilingual models like XLM-RoBERTa achieve high CLS similarity between translation pairs, but this similarity doesn't effectively discriminate translations from random pairs. We hypothesize that **explicit alignment training** will create a more meaningful semantic space, improving performance on downstream cross-lingual tasks.

## 🏗️ Architecture

### Input Structure

Concatenated bilingual sequences:
```
[CLS] lang1_tokens [SEP] [CLS] lang2_tokens [SEP]
```

### Training Objectives

**1. Standard MLM (Pass 1)**
- Randomly mask ~15% of tokens across both languages
- Model predicts masked tokens
- Loss: Cross-entropy (standard BERT MLM)

**2. CLS Alignment (Passes 2a & 2b)**

*Pass 2a:* Mask all lang2 tokens + second CLS
- Extract CLS₁ from first position (represents lang1)

*Pass 2b:* Mask all lang1 tokens + first SEP
- Extract CLS₂ from second CLS position (represents lang2)

*Alignment Loss:* Minimize distance between CLS₁ and CLS₂
- Options: MSE, cosine similarity loss, or contrastive loss

### Combined Loss

```
L_total = L_mlm + λ × L_align
```

Where:
- `L_mlm` = Masked language modeling loss
- `L_align` = Alignment loss between CLS₁ and CLS₂
- `λ` = Hyperparameter balancing the two objectives (default: 1.0)

## 📊 Key Innovation

Unlike standard multilingual pre-training, we **explicitly train** the model to align CLS representations across languages. This should produce:

1. **Better semantic alignment** - CLS tokens capture language-agnostic meaning
2. **Improved discrimination** - Model distinguishes translations from random pairs
3. **Higher-quality embeddings** - More useful for downstream cross-lingual tasks

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/llm-jepa-translation.git
cd llm-jepa-translation

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Dataset: OPUS-100

This project uses the **OPUS-100** dataset ([Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)), which provides:
- 100 language pairs (all English-centric)
- Up to 1M training examples per pair
- Validation and test splits
- Wide language diversity (European, Asian, African languages)

**Explore available language pairs:**
```bash
# List all available pairs
python scripts/list_opus100_pairs.py

# Filter for specific language (e.g., English pairs)
python scripts/list_opus100_pairs.py --filter en

# Show dataset sizes (slower, downloads metadata)
python scripts/list_opus100_pairs.py --sizes
```

**Test dataset loading:**
```bash
python scripts/test_opus100_loading.py
```

### Training

Train the model with dual objectives:

```bash
python scripts/train.py --config experiments/exp_test/config.yaml
```

**Configuration options** (`config.yaml`):
- `base_model`: Pre-trained model (default: `xlm-roberta-base`)
- `lambda_alignment`: Weight for alignment loss (default: 1.0)
- `alignment_loss_type`: Type of alignment loss (`mse`, `cosine`, `contrastive`)
- `mlm_probability`: MLM masking probability (default: 0.15)
- `lang_pairs`: Language pairs from OPUS-100 (e.g., `["de-en", "fr-en", "en-ja"]`)
- `max_examples_per_pair`: Maximum examples per language pair (uses `min(max_examples, available)`)
  - Set to specific number (e.g., `50000`) to limit dataset size
  - Set to `null` to load all available examples

**Resume training:**
```bash
python scripts/train.py --config config.yaml --resume checkpoints/checkpoint_epoch_5.pt
```

### Evaluation

Evaluate alignment quality, discrimination ability, and retrieval accuracy:

```bash
python scripts/evaluate.py \
    --config experiments/exp_test/config.yaml \
    --checkpoint experiments/exp_test/checkpoints/best_model \
    --output results.json
```

**Evaluation metrics:**
- **CLS Cosine Similarity**: Alignment quality between translation pairs
- **Discrimination Score**: Difference between translation and random pair similarities
- **Retrieval Accuracy**: Cross-lingual retrieval performance (top-k)

**Evaluate specific language pair:**
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model \
    --lang_pair de-en \
    --max_examples 500
```

## 📁 Project Structure

```
├── src/
│   ├── models/
│   │   └── bert_dual_objective.py   # Main model with dual objectives
│   ├── data/
│   │   ├── datasets.py               # Dataset loading and concatenation
│   │   ├── collators.py              # Batch collation with 3-way masking
│   │   └── masking.py                # MLM and cross-lingual masking
│   └── training/
│       ├── trainer.py                # Training loop
│       └── metrics.py                # Evaluation metrics
├── scripts/
│   ├── train.py                      # Training script
│   └── evaluate.py                   # Evaluation script
├── experiments/
│   └── exp_test/
│       ├── config.yaml               # Configuration file
│       └── checkpoints/              # Model checkpoints
└── requirements.txt
```

## 🔬 Research Questions

1. **Does explicit alignment improve semantic quality?**
   - Compare CLS similarity distributions before/after training
   - Measure discrimination ability (translation vs random pairs)

2. **What is the optimal balance between objectives?**
   - Ablation study on λ values
   - MLM-only vs alignment-only vs combined training

3. **Does this transfer to downstream tasks?**
   - Cross-lingual sentence retrieval
   - Semantic textual similarity
   - Zero-shot cross-lingual transfer

4. **How does it compare to contrastive approaches?**
   - Compare MSE, cosine, and contrastive alignment losses

## 📈 Expected Improvements

Based on preliminary analysis (see `roberta.ipynb`), pre-trained XLM-RoBERTa shows:
- **High CLS similarity** (~0.89) but **poor discrimination** (~0.0)
- Cannot distinguish translation pairs from random pairs

With explicit alignment training, we expect:
- **Improved discrimination** (>0.05 difference)
- **More meaningful similarity scores**
- **Better retrieval accuracy** (>10% improvement)

## 🔧 Development

### Running Tests

```bash
# Create test directory
mkdir -p tests

# Run tests (after implementing)
python -m pytest tests/
```

### Analyzing Results

See `roberta.ipynb` for analysis methods:
- CLS similarity distributions
- Discrimination tests
- Cross-lingual retrieval evaluation

## 📖 Citation

This work is inspired by:

1. **JEPA (Joint Embedding Predictive Architecture)**
   - LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence

2. **LLM-JEPA**
   - Recent work applying JEPA principles to language models

3. **XLM-RoBERTa**
   - Conneau et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional evaluation metrics
- New alignment loss functions
- Downstream task integration
- Multi-GPU training support

## 📝 License

MIT License

## 🙏 Acknowledgments

- Brown University CS2952X
- HuggingFace Transformers library
- OPUS-100 dataset (Helsinki-NLP)
- OPUS corpus (Jörg Tiedemann)
