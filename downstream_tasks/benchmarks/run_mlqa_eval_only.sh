#!/bin/bash
#SBATCH --job-name=mlqa_eval
#SBATCH --output=logs/mlqa_eval_%j.out
#SBATCH --error=logs/mlqa_eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# =============================================================================
# MLQA Evaluation Only (Skip Training) - Full G-XLT Matrix
# =============================================================================
#
# This script runs MLQA evaluation on an already trained QA model.
# Evaluates the full 7x7 G-XLT matrix (49 language pair combinations)
# matching the paper's Table 6 and Table 9 format.
#
# Usage:
#   sbatch run_mlqa_eval_only.sh <trained_model_path>
#
# Examples:
#   sbatch run_mlqa_eval_only.sh ./results/mlqa_xlm-roberta-base
#   sbatch run_mlqa_eval_only.sh your-org/your-qa-model
#
# =============================================================================

# Print job information
echo "=========================================="
echo "MLQA Evaluation Only"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# --- Navigate to project directory FIRST ---
cd /users/tgillin/files/llm-jepa-translation

# Create logs directory if it doesn't exist
mkdir -p logs

# --- Configuration ---
TRAINED_MODEL="${1:-./results/mlqa_xlm-roberta-base}"
MODEL_NAME="${2:-xlm-roberta-base}"
OUTPUT_CSV="downstream_tasks/benchmarks/mlqa_eval_results_${SLURM_JOB_ID:-local}.csv"

# --- Activate virtual environment ---
echo "Activating virtual environment..."
source venv/bin/activate

# Set PyTorch memory allocator
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- Verify GPU availability ---
echo ""
echo "=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi
echo ""

# --- Environment variables ---
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Print Configuration ---
echo "=========================================="
echo "Configuration"
echo "=========================================="
echo "  Trained model: $TRAINED_MODEL"
echo "  Model name (for CSV): $MODEL_NAME"
echo "  Output CSV: $OUTPUT_CSV"
echo "  Working directory: $(pwd)"
echo "=========================================="
echo ""

# --- Run Evaluation Only ---
echo "Starting MLQA evaluation (skip training)..."
echo ""

python downstream_tasks/benchmarks/benchmark_mlqa.py \
    --model "$MODEL_NAME" \
    --skip_training \
    --trained_model_path "$TRAINED_MODEL" \
    --languages all \
    --full_matrix \
    --output_csv "$OUTPUT_CSV"

EVAL_EXIT_CODE=$?

# --- Completion ---
echo ""
echo "=========================================="
echo "MLQA Evaluation Summary"
echo "=========================================="
echo "Exit Code: $EVAL_EXIT_CODE"
echo "End Time: $(date)"
echo ""

# Display results
if [ -f "$OUTPUT_CSV" ]; then
    echo "Results:"
    echo "----------------------------------------"
    cat "$OUTPUT_CSV"
    echo ""
    echo "----------------------------------------"
    echo "Results saved to: $OUTPUT_CSV"
fi

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ MLQA EVALUATION COMPLETED SUCCESSFULLY"
else
    echo ""
    echo "❌ MLQA EVALUATION FAILED"
fi

echo "=========================================="

exit $EVAL_EXIT_CODE


