#!/bin/bash
#SBATCH --job-name=jepa_train
#SBATCH --output=logs/training_output_%j.log
#SBATCH --error=logs/training_error_%j.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Navigate to project directory
cd /users/tgillin/files/llm-jepa-translation

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Set up Weights & Biases API key
if [ -f ~/.wandb_api_key ]; then
    export WANDB_API_KEY=$(cat ~/.wandb_api_key)
    echo "✓ W&B API key loaded from ~/.wandb_api_key"
elif [ -z "$WANDB_API_KEY" ]; then
    echo "⚠ Warning: WANDB_API_KEY not set. W&B logging may fail."
    echo "  To fix: Run 'wandb login' or create ~/.wandb_api_key"
fi

# Verify GPU availability
echo ""
echo "=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi
echo ""

# Set environment variables for better performance
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# W&B settings for better cluster compatibility
export WANDB_CONSOLE=off  # Disable interactive features
export WANDB_DIR=/users/tgillin/files/llm-jepa-translation/wandb  # Set wandb directory

# Parse command line arguments
CONFIG_FILE="${1:-experiments/exp_test/config.yaml}"
RESUME_CHECKPOINT="${2:-}"

echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resuming from: $RESUME_CHECKPOINT"
fi
echo ""

# Run training
echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo ""

if [ -n "$RESUME_CHECKPOINT" ]; then
    python scripts/train.py --config "$CONFIG_FILE" --resume "$RESUME_CHECKPOINT"
else
    python scripts/train.py --config "$CONFIG_FILE"
fi

TRAINING_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training Summary"
echo "=========================================="
echo "Exit Code: $TRAINING_EXIT_CODE"
echo "End Time: $(date)"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✓ TRAINING COMPLETED SUCCESSFULLY"
else
    echo "❌ TRAINING FAILED"
fi
echo "=========================================="

exit $TRAINING_EXIT_CODE

