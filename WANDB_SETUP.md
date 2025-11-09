# Weights & Biases Setup Guide

This guide explains how to set up and use Weights & Biases (W&B) for experiment tracking in this project.

## Initial Setup

### 1. Create a W&B Account
1. Go to [https://wandb.ai/signup](https://wandb.ai/signup)
2. Sign up (free for academics and personal projects)
3. Verify your email

### 2. Get Your API Key
1. Log in to W&B
2. Go to [https://wandb.ai/authorize](https://wandb.ai/authorize)
3. Copy your API key

### 3. Set Up API Key on the Cluster

**Option A: Interactive Setup (Recommended for First Time)**
```bash
# Activate your environment
source venv/bin/activate

# Run wandb login (this will prompt for your API key)
wandb login
```

**Option B: Environment Variable**
```bash
# Add to your ~/.bashrc or set before running
export WANDB_API_KEY="your-api-key-here"
```

**Option C: Store in a file** (used by our Slurm script)
```bash
# Create a secure file with your API key
echo "your-api-key-here" > ~/.wandb_api_key
chmod 600 ~/.wandb_api_key
```

## Using W&B with Training

### Normal Training with W&B
```bash
sbatch run_training.sh experiments/exp_test/config_small.yaml
```

W&B will automatically:
- Track all hyperparameters from your config
- Log training metrics (loss, accuracy, etc.) in real-time
- Save model checkpoints as artifacts
- Create interactive plots
- Track system metrics (GPU usage, CPU, memory)

### Disable W&B (for testing)
```bash
python scripts/train.py --config experiments/exp_test/config.yaml --no-wandb
```

Or in the config file:
```yaml
wandb:
  enabled: false
```

## What Gets Logged

### Hyperparameters
All configuration values from your YAML config:
- Model architecture settings
- Training hyperparameters
- Dataset configuration
- Learning rates, batch sizes, etc.

### Training Metrics (Real-time)
Logged every `log_interval` steps:
- `train/total_loss`: Combined MLM + alignment loss
- `train/mlm_loss`: Masked language modeling loss
- `train/alignment_loss`: Cross-lingual alignment loss
- `train/weighted_alignment_loss`: Lambda * alignment loss
- `train/cls_cosine_sim`: CLS token similarity
- `train/mlm_accuracy`: MLM prediction accuracy
- `train/learning_rate`: Current learning rate

### Validation Metrics (Each Epoch)
- `val/total_loss`: Validation loss
- `val/mlm_loss`: Validation MLM loss
- `val/alignment_loss`: Validation alignment loss
- `val/cls_cosine_sim`: Validation CLS similarity

### Model Artifacts
Automatically saved to W&B:
- `best_model.pt`: Best model by validation loss
- `final_model.pt`: Final model after all epochs

### System Metrics
Automatically tracked:
- GPU utilization
- GPU memory usage
- CPU usage
- System memory
- Disk I/O

## Viewing Your Results

### During Training
1. Check the training log for the W&B URL:
   ```
   ✓ W&B initialized: test-run-small
     URL: https://wandb.ai/your-username/llm-jepa-translation/runs/xyz123
   ```

2. Open the URL in your browser to see live training progress

### Dashboard Features
- **Overview**: Key metrics and system info
- **Charts**: Interactive plots of all metrics
- **System**: GPU/CPU utilization over time
- **Model**: Saved model artifacts
- **Logs**: Complete training logs
- **Files**: All saved files and configs

## Comparing Experiments

### View All Runs
Go to your project page:
```
https://wandb.ai/your-username/llm-jepa-translation
```

### Compare Multiple Runs
1. Select multiple runs (checkbox on left)
2. Click "Compare" or use the parallel coordinates plot
3. See side-by-side metrics, parameters, and results

### Useful Comparisons
- Different lambda values (alignment weight)
- Different alignment loss types (MSE, cosine, contrastive)
- Different learning rates
- Different language pairs

## Configuration Tips

### Custom Run Name
In your config file:
```yaml
wandb:
  run_name: "lambda-1.0-mse-lr2e-5"  # Descriptive name
```

### Tags for Organization
```yaml
wandb:
  tags: ["xlm-roberta-base", "lambda-1.0", "mse-loss", "de-en"]
```

### Add Notes
```yaml
wandb:
  notes: "Testing higher lambda value with MSE loss on German-English pairs"
```

### Group Related Runs
```yaml
wandb:
  group: "lambda-ablation"  # Groups runs together
```

## Troubleshooting

### "wandb: ERROR Not logged in"
Solution:
```bash
source venv/bin/activate
wandb login
# Enter your API key when prompted
```

### API Key Issues with Slurm
Make sure the API key file exists:
```bash
ls -la ~/.wandb_api_key
```

Or set the environment variable in your Slurm script:
```bash
export WANDB_API_KEY="your-api-key-here"
```

### "wandb: WARNING Run xxxxxx not found"
This is normal when resuming - W&B will create a new run if the old one doesn't exist.

### Disable W&B for Debugging
```bash
export WANDB_MODE=disabled
# or
python scripts/train.py --no-wandb
```

### Offline Mode (No Internet)
```bash
export WANDB_MODE=offline
# Sync later with: wandb sync
```

## Best Practices

### 1. Meaningful Run Names
Use descriptive names that capture key hyperparameters:
- ❌ Bad: "run-1", "test", "final"
- ✅ Good: "lambda1.0-mse-de-en", "lr5e-5-batch32"

### 2. Consistent Tags
Use tags to organize experiments:
- Model type: `xlm-roberta-base`, `xlm-roberta-large`
- Loss type: `mse-loss`, `cosine-loss`
- Language pairs: `de-en`, `fr-en`, `multi-lang`
- Experiment type: `baseline`, `ablation`, `final`

### 3. Document Experiments
Add notes about:
- What you're testing
- Why you chose these hyperparameters
- Expected outcomes
- Any issues or observations

### 4. Use Groups for Ablations
Group related experiments:
```yaml
wandb:
  group: "lambda-sweep"
  tags: ["ablation", "lambda"]
```

### 5. Review Runs Regularly
- Archive failed runs
- Star successful runs
- Add notes about insights

## Advanced Features

### Sweeps (Hyperparameter Search)
Create a sweep config:
```yaml
# sweep_config.yaml
program: scripts/train.py
method: bayes
metric:
  name: val/total_loss
  goal: minimize
parameters:
  lambda_alignment:
    values: [0.1, 0.5, 1.0, 2.0, 5.0]
  learning_rate:
    values: [1e-5, 2e-5, 5e-5]
```

Run sweep:
```bash
wandb sweep sweep_config.yaml
wandb agent your-username/llm-jepa-translation/sweep-id
```

### Custom Charts
Add custom visualizations in the W&B dashboard:
- Click "Add Panel"
- Choose chart type
- Select metrics to visualize

### Download Models
```python
import wandb

api = wandb.Api()
artifact = api.artifact('your-username/llm-jepa-translation/model-xyz:latest')
artifact.download()
```

## Resources

- [W&B Documentation](https://docs.wandb.ai/)
- [W&B Python Library](https://docs.wandb.ai/ref/python)
- [Example Projects](https://wandb.ai/gallery)
- [Community Forum](https://community.wandb.ai/)

## Support

- W&B Issues: [GitHub Issues](https://github.com/wandb/wandb/issues)
- Project Issues: See main README

