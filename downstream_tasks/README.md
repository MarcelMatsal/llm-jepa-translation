# Downstream Task Testing

Quick testing suite to verify that the trained dual-objective model maintains BERT-compatible architecture and can be used for standard downstream tasks.

## 🎯 Purpose

These scripts allow you to:
1. **Verify architecture** - Confirm the model is BERT-compatible
2. **Test embeddings** - Extract and compare sentence embeddings
3. **Test MLM** - Verify masked language modeling still works
4. **Prepare for real tasks** - Ensure the model is ready for downstream fine-tuning

## 📁 Files

- `load_model.py` - Load model and verify architecture
- `test_embeddings.py` - Test sentence embedding extraction and similarity
- `test_mlm.py` - Test masked language modeling capabilities
- `README.md` - This file

## 🚀 Quick Start

### Using SLURM (Recommended for Cluster)

Submit a batch job to test your model:

```bash
# Test the best model checkpoint
sbatch run_downstream_tests.sh

# Test a specific checkpoint
sbatch run_downstream_tests.sh experiments/exp_test/checkpoints/checkpoint_epoch_5

# Force CPU testing (if GPU unavailable)
sbatch run_downstream_tests.sh experiments/exp_test/checkpoints/best_model cpu
```

**What it does:**
- Allocates GPU resources
- Runs all three tests sequentially
- Saves results to `logs/downstream_output_[job_id].log`
- Returns exit code 0 if all tests pass

**Check job status:**
```bash
squeue -u $USER  # See running jobs
cat logs/downstream_output_*.log  # View latest results
```

### Interactive Testing (Local or Interactive Node)

### 1. Load and Verify Model

```bash
python downstream_tasks/load_model.py \
    --checkpoint experiments/exp_test/checkpoints/best_model \
    --device cuda
```

**What it checks:**
- ✓ Model architecture components
- ✓ Model dimensions and configuration
- ✓ Tokenizer functionality
- ✓ Forward pass works
- ✓ CLS embedding extraction works

### 2. Test Sentence Embeddings

```bash
python downstream_tasks/test_embeddings.py \
    --checkpoint experiments/exp_test/checkpoints/best_model \
    --device cuda
```

**What it tests:**
- Extracts CLS embeddings for test sentences
- Computes similarity matrix
- Shows cross-lingual alignment (English ↔ German ↔ French)
- Demonstrates semantic similarity vs. dissimilarity

**Example output:**
```
SENTENCE SIMILARITY TEST
============================================================

Extracting embeddings...
  ✓ en1: The cat sits on the mat.
  ✓ en2: A cat is sitting on a mat.
  ✓ de1: Die Katze sitzt auf der Matte.
  
Similarity Matrix:
           en1       en2       en3       de1       fr1
      en1  1.0000    0.9234    0.4532    0.8765    0.8634
      en2  0.9234    1.0000    0.4321    0.8543    0.8421
      ...
```

**Interactive mode** (optional):
```bash
python downstream_tasks/test_embeddings.py \
    --checkpoint experiments/exp_test/checkpoints/best_model \
    --device cuda \
    --custom
```

### 3. Test Masked Language Modeling

```bash
python downstream_tasks/test_mlm.py \
    --checkpoint experiments/exp_test/checkpoints/best_model \
    --device cuda
```

**What it tests:**
- MLM predictions on English sentences
- Multilingual MLM (German, French, Spanish)
- Verifies the MLM head still works correctly

**Example output:**
```
MASKED LANGUAGE MODELING TEST - ENGLISH
============================================================

Input: The cat sits on the <mask>.
  Predictions:
    mat                  0.3456
    floor                0.2134
    table                0.1543
    ground               0.0987
    carpet               0.0765
```

**Interactive mode** (optional):
```bash
python downstream_tasks/test_mlm.py \
    --checkpoint experiments/exp_test/checkpoints/best_model \
    --device cuda \
    --custom
```

## 💡 Usage Examples

### Quick Architecture Check

```bash
# Just verify the model loads and has correct architecture
python downstream_tasks/load_model.py \
    --checkpoint experiments/exp_test/checkpoints/checkpoint_epoch_3
```

### Check Cross-Lingual Alignment Quality

```bash
# See how well your trained model aligns languages
python downstream_tasks/test_embeddings.py \
    --checkpoint experiments/exp_test/checkpoints/best_model
```

### Verify MLM Capability Preserved

```bash
# Ensure training didn't break the MLM head
python downstream_tasks/test_mlm.py \
    --checkpoint experiments/exp_test/checkpoints/best_model
```

## 🔍 What You're Testing

### Architecture Compatibility

The trained model should be **fully compatible** with standard BERT/XLM-RoBERTa because:

1. **Base Architecture**: XLM-RoBERTa transformer (unchanged)
2. **MLM Head**: Standard masked language modeling head (unchanged)
3. **CLS Embeddings**: Standard CLS token extraction (enhanced by training)

### What's Different

The **training process** explicitly aligned CLS tokens across languages, so:
- CLS embeddings should be more semantically meaningful
- Cross-lingual similarities should be higher for translations
- Discrimination between translations and non-translations should be better

### Expected Results

**Good signs:**
- ✓ Architecture checks all pass
- ✓ Cross-lingual translation pairs have high similarity (>0.8)
- ✓ Non-translation pairs have lower similarity
- ✓ MLM predictions make sense
- ✓ Similar sentences (same language) have high similarity

**Potential issues:**
- ⚠️ All similarities are very high (poor discrimination)
- ⚠️ Cross-lingual similarities lower than expected
- ⚠️ MLM predictions nonsensical (training broke something)

## 🎓 Next Steps

Once you've verified the architecture:

1. **Fine-tune on downstream tasks:**
   - Sentiment analysis
   - Named entity recognition
   - Question answering
   - Text classification

2. **Use as sentence encoder:**
   ```python
   from src.models.bert_dual_objective import BertDualObjective
   
   model = BertDualObjective.from_pretrained("path/to/checkpoint")
   embeddings = model.extract_cls_embeddings(input_ids, attention_mask, cls_positions)
   ```

3. **Access underlying XLM-RoBERTa:**
   ```python
   xlm_roberta = model.mlm_model  # Standard XLMRobertaForMaskedLM
   # Use with HuggingFace Trainer or any other framework
   ```

4. **Evaluate on benchmarks:**
   - XNLI (cross-lingual NLI)
   - BUCC (bilingual retrieval)
   - Tatoeba (translation pair retrieval)

## 🔧 Troubleshooting

### Model won't load
```bash
# Check that checkpoint directory contains:
ls -la experiments/exp_test/checkpoints/best_model/
# Should have: config.json, pytorch_model.bin, tokenizer files, dual_objective_config.json
```

### CUDA out of memory
```bash
# Use CPU instead
python downstream_tasks/load_model.py --checkpoint path/to/model --device cpu
```

### Import errors
```bash
# Make sure you're running from the project root
cd /users/tgillin/files/llm-jepa-translation
python downstream_tasks/load_model.py --checkpoint ...
```

## 📊 Comparing to Baseline

To see the improvement from your training:

1. Load a pre-trained XLM-RoBERTa (no dual-objective training):
   ```bash
   # Use xlm-roberta-base as checkpoint
   python downstream_tasks/test_embeddings.py --checkpoint xlm-roberta-base
   ```

2. Compare similarities:
   - Your model should have **better discrimination**
   - Translation pairs should have **higher relative similarity** vs. random pairs

## 📝 Notes

- These are **quick tests**, not full benchmarks
- For production evaluation, use standard benchmarks (XNLI, etc.)
- The scripts are designed to be simple and easy to modify
- Add your own test cases as needed!

