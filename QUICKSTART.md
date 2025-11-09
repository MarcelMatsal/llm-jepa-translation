# Quick Start Guide

## Test the Implementation

### 1. Run Tests

Test the data pipeline:
```bash
python tests/test_data.py
```

Test the model:
```bash
python tests/test_model.py
```

Both should output "ALL TESTS PASSED ✓"

## Train the Model

### 2. Quick Test Training (Small Dataset)

For a quick test run with limited data:

```bash
python scripts/train.py --config experiments/exp_test/config.yaml
```

This will:
- Load XLM-RoBERTa base model
- Load WMT19 de-en, fr-en, cs-en datasets
- Train with dual objectives (MLM + alignment)
- Save checkpoints to `experiments/exp_test/checkpoints/`

### 3. Evaluate the Model

Evaluate a trained model:

```bash
python scripts/evaluate.py \
    --config experiments/exp_test/config.yaml \
    --checkpoint experiments/exp_test/checkpoints/best_model \
    --output results.json
```

This will compute:
- CLS cosine similarity
- Discrimination score (translation vs random pairs)
- Retrieval accuracy

### 4. Compare with Baseline

Evaluate pre-trained XLM-RoBERTa (without our training):

```bash
python scripts/evaluate.py \
    --config experiments/exp_test/config.yaml \
    --output baseline_results.json
```

Note: Without `--checkpoint`, it evaluates the pre-trained model.

## Configuration Options

Edit `experiments/exp_test/config.yaml` to customize:

```yaml
model:
  lambda_alignment: 1.0      # Try 0.5, 2.0, 5.0
  alignment_loss_type: "mse" # Try "cosine", "contrastive"
  mlm_probability: 0.15

data:
  lang_pairs: ["de-en"]      # Start with one pair for quick testing
  batch_size: 8              # Reduce if GPU memory is limited
  max_examples_per_pair: 1000  # Limit data for quick experiments

training:
  epochs: 3                  # Fewer epochs for quick testing
  learning_rate: 2e-5
```

## Troubleshooting

### GPU Memory Issues

If you get CUDA out of memory errors:
1. Reduce `batch_size` in config (try 4 or 8)
2. Reduce `max_length` (try 128)
3. Increase `accumulation_steps` to 2 or 4

### DataLoader Errors

If you get multiprocessing errors:
1. Set `num_workers: 0` in config

### Download Issues

If WMT19 datasets fail to download:
1. Try a different language pair
2. Check your internet connection
3. Try setting a cache directory: `export HF_DATASETS_CACHE=/path/to/cache`

## Expected Training Time

On GPU (e.g., V100):
- 1 epoch, 1000 examples per pair, 3 pairs: ~10-15 minutes
- Full training (10 epochs, all data): several hours

On CPU:
- Much slower, recommended for testing only

## Next Steps

1. **Run baseline evaluation** to see pre-trained performance
2. **Train for 1-2 epochs** with small dataset
3. **Evaluate** and check if discrimination improves
4. **Run ablations**:
   - Train with λ=0 (MLM only)
   - Train with different λ values
   - Try different loss types
5. **Analyze results** in notebook

## Useful Commands

**Monitor GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Resume training:**
```bash
python scripts/train.py \
    --config experiments/exp_test/config.yaml \
    --resume experiments/exp_test/checkpoints/checkpoint_epoch_5.pt
```

**Evaluate specific language pair:**
```bash
python scripts/evaluate.py \
    --checkpoint experiments/exp_test/checkpoints/best_model \
    --lang_pair de-en \
    --max_examples 500
```

## Understanding Results

Good results show:
- **CLS Similarity**: ~0.85-0.90 (maintained from baseline)
- **Discrimination Score**: >0.05 (improvement from ~0.0)
- **Retrieval Accuracy**: >0.7 for @1

If discrimination doesn't improve:
- Try increasing λ
- Try longer training
- Try contrastive loss instead of MSE

## Questions?

Check the full README.md for detailed documentation.

