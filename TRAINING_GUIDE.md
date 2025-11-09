# Training Guide

This guide explains how to train your dual-objective BERT model using Slurm.

## Quick Start

### 1. Test Run (Recommended First!)
Start with a small test to ensure everything works:

```bash
sbatch run_training.sh experiments/exp_test/config_small.yaml
```

This will:
- Train on only 1,000 German-English examples
- Run for 2 epochs (~10-15 minutes)
- Use smaller batch size and sequence length
- Save checkpoints to `experiments/exp_test/checkpoints_small/`

### 2. Full Training
Once the test run succeeds, launch full training:

```bash
sbatch run_training.sh experiments/exp_test/config.yaml
```

This will:
- Train on German-English, French-English, and Czech-English
- Run for 10 epochs (~several hours)
- Use full dataset (can take a while to load)
- Save checkpoints to `experiments/exp_test/checkpoints/`

### 3. Resume Training
If training is interrupted, resume from a checkpoint:

```bash
sbatch run_training.sh experiments/exp_test/config.yaml experiments/exp_test/checkpoints/checkpoint_epoch_5.pt
```

## Configuration Files

### Small Test Config (`config_small.yaml`)
- **Purpose**: Quick testing and validation
- **Dataset**: 1,000 de-en examples
- **Training time**: ~10-15 minutes
- **Epochs**: 2
- **Batch size**: 8 (effective: 16 with accumulation)

### Full Config (`config.yaml`)
- **Purpose**: Production training
- **Dataset**: Full de-en, fr-en, cs-en
- **Training time**: Several hours
- **Epochs**: 10
- **Batch size**: 16

## Resource Allocation

The training script requests:
- **GPU**: 1 GPU (adjust with `#SBATCH --gres=gpu:N`)
- **Memory**: 64GB RAM
- **CPUs**: 8 cores
- **Time**: 24 hours (adjust with `#SBATCH --time=HH:MM:SS`)
- **Partition**: gpu (adjust to your cluster's GPU partition name)

## Monitoring Training

### Check Job Status
```bash
squeue -u $USER
```

### View Live Output
```bash
tail -f training_output_<JOB_ID>.log
```

### View Errors
```bash
tail -f training_error_<JOB_ID>.log
```

### Cancel Job
```bash
scancel <JOB_ID>
```

## Key Hyperparameters to Tune

### Lambda (Alignment Weight)
In your config file, adjust:
```yaml
model:
  lambda_alignment: 1.0  # Try: 0.1, 0.5, 1.0, 2.0
```

### Alignment Loss Type
```yaml
model:
  alignment_loss_type: "mse"  # Try: 'mse', 'cosine', 'contrastive'
```

### Learning Rate
```yaml
training:
  learning_rate: 2e-5  # Try: 1e-5, 2e-5, 5e-5
```

### Batch Size
```yaml
data:
  batch_size: 16  # Adjust based on GPU memory
training:
  accumulation_steps: 1  # Effective batch = batch_size * accumulation_steps
```

## Expected Behavior

### During Training You Should See:
- MLM loss decreasing over time
- Alignment loss decreasing
- CLS cosine similarity increasing (should go above 0.9)
- Regular checkpoint saves

### Warning Signs:
- MLM loss not decreasing → learning rate too low/high
- Alignment loss = 0 → lambda too low or masking issue
- OOM errors → reduce batch size or sequence length
- Very slow progress → reduce dataset size or increase num_workers

## Output Structure

After training, you'll have:
```
experiments/exp_test/
├── checkpoints/
│   ├── checkpoint_epoch_1.pt
│   ├── checkpoint_epoch_2.pt
│   ├── ...
│   ├── best_model.pt
│   └── final_model.pt
└── training_logs.json
```

## Next Steps After Training

1. **Evaluate the model** using `scripts/evaluate.py`
2. **Compare to baseline** (pre-trained XLM-RoBERTa)
3. **Analyze metrics**: CLS similarity, discrimination, retrieval
4. **Iterate**: Adjust hyperparameters based on results

## Troubleshooting

### Job Pending for Long Time
- Check queue: `squeue`
- Check partition availability: `sinfo`
- Try different partition if available

### CUDA Out of Memory
- Reduce `batch_size` in config
- Reduce `max_length` in config
- Increase `accumulation_steps` to maintain effective batch size

### Dataset Loading Takes Forever
- Use `max_examples_per_pair` to limit dataset size
- Increase `num_workers` for faster data loading
- Consider caching dataset to disk

### Training is Very Slow
- Check GPU utilization: `nvidia-smi`
- Increase `num_workers` in config
- Reduce sequence length if very long

## Example: Quick Ablation Study

Create multiple configs to test different lambda values:

```bash
# Copy base config
cp experiments/exp_test/config_small.yaml experiments/exp_test/config_lambda_0.1.yaml
cp experiments/exp_test/config_small.yaml experiments/exp_test/config_lambda_1.0.yaml
cp experiments/exp_test/config_small.yaml experiments/exp_test/config_lambda_2.0.yaml

# Edit each file to change lambda_alignment, then submit:
sbatch run_training.sh experiments/exp_test/config_lambda_0.1.yaml
sbatch run_training.sh experiments/exp_test/config_lambda_1.0.yaml
sbatch run_training.sh experiments/exp_test/config_lambda_2.0.yaml
```

## Contact & Support

For issues specific to your Slurm cluster setup, consult your cluster documentation or ask your system administrator about:
- Available GPU partitions
- GPU types and availability
- Time limits and quotas
- Recommended resource allocations

