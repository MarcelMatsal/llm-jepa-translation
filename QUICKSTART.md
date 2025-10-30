# Quick Start Guide

## ✅ Basic Test Passed!

All core components are working:
- Model creation ✓
- Forward pass ✓
- Loss computation ✓
- EMA updates ✓
- Training steps ✓

## Running the Basic Test

```bash
# Activate virtual environment
source venv/bin/activate

# Run basic test
python scripts/test_basic.py
```

## Next Steps: Training on Real Data

### Option 1: Small HuggingFace Dataset (Recommended for First Test)

```bash
# Try with a small dataset first
python scripts/train.py \
    --dataset opus_books \
    --lang_pair en-fr \
    --epochs 1 \
    --batch_size 8 \
    --encoder_name bert-base-multilingual-cased
```

**Note**: This will download the dataset on first run (may take a few minutes).

### Option 2: Create Your Own Small Dataset

Create a simple JSONL file `data/test.jsonl`:

```json
{"text_src": "Hello world", "text_tgt": "Bonjour le monde", "lang_src": "en", "lang_tgt": "fr"}
{"text_src": "How are you?", "text_tgt": "Comment allez-vous?", "lang_src": "en", "lang_tgt": "fr"}
```

Then modify the training script to use it, or add support for JSONL files.

### Option 3: Use Synthetic Data (Already Working)

The test script already creates synthetic data - you can extend it for quick experiments.

## Monitoring Training

Watch for:
- **Loss decreasing**: Should decrease over epochs
- **Cosine similarity**: Should increase (from negative to positive)
- **Embedding diversity**: Should stay high (prevent collapse)
- **Linearity error**: Should decrease (indicates linear transformation)

## Troubleshooting

### Memory Issues
- Reduce `batch_size` (try 4 or 8)
- Use smaller encoder: `--encoder_name distilbert-base-multilingual-cased`

### Slow Training
- This is normal - transformer encoders are computationally expensive
- Consider using GPU if available: `--device cuda`

### Dataset Loading Issues
- Some HuggingFace datasets require specific configuration
- Check dataset documentation for correct format

## Quick Experiment Ideas

1. **Test different pooling strategies**:
   ```bash
   python scripts/train.py --pooling mean  # instead of cls
   ```

2. **Test different EMA decay rates**:
   Modify `tau` in the model (default 0.999)

3. **Test different languages**:
   Try different language pairs: `--lang_pair en-de` or `--lang_pair fr-es`

## File Structure

- `scripts/test_basic.py` - Basic functionality test ✓
- `scripts/train.py` - Main training script
- `scripts/evaluate.py` - Evaluation script
- `src/models/` - Model architectures
- `src/data/` - Dataset loaders
- `src/training/` - Training utilities

## Need Help?

Check the architecture documentation:
- `ARCHITECTURE.md` - Detailed architecture explanation
- `IMPLEMENTATION_PLAN.md` - Implementation details

