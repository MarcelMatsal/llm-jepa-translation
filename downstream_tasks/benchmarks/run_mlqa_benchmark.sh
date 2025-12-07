#!/bin/bash
#SBATCH --job-name=mlqa_benchmark
#SBATCH --output=logs/mlqa_%j.out
#SBATCH --error=logs/mlqa_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# =============================================================================
# MLQA Cross-lingual QA Benchmark
# =============================================================================
#
# This script runs the full MLQA evaluation following the paper methodology:
# 1. Train on English SQuAD v1.1
# 2. Evaluate zero-shot on all MLQA language pairs (monolingual + cross-lingual)
# 3. Report F1 and Exact Match scores
#
# Usage:
#   sbatch downstream_tasks/benchmarks/run_mlqa_benchmark.sh <model_name>
#
# Examples:
#   sbatch downstream_tasks/benchmarks/run_mlqa_benchmark.sh xlm-roberta-base
#   sbatch downstream_tasks/benchmarks/run_mlqa_benchmark.sh your-org/your-custom-model
#
# =============================================================================

# Print job information
echo "=========================================="
echo "MLQA Cross-lingual QA Benchmark"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# --- Navigate to project directory FIRST ---
cd /users/tgillin/files/llm-jepa-translation

# Create logs directory if it doesn't exist (now we're in the right place)
mkdir -p logs

# --- Configuration ---
MODEL="${1:-xlm-roberta-base}"
TRAIN_EPOCHS="${2:-2}"
BATCH_SIZE="${3:-12}"
LEARNING_RATE="${4:-3e-5}"
OUTPUT_CSV="downstream_tasks/benchmarks/mlqa_results_${SLURM_JOB_ID:-local}.csv"

# --- Activate virtual environment ---
echo "Activating virtual environment..."
source venv/bin/activate

# Set PyTorch memory allocator to prevent fragmentation
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
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# --- Print Configuration ---
echo "=========================================="
echo "Configuration"
echo "=========================================="
echo "  Model: $MODEL"
echo "  Training epochs: $TRAIN_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Output CSV: $OUTPUT_CSV"
echo "  Working directory: $(pwd)"
echo "=========================================="
echo ""

# --- Run Benchmark ---
echo "Starting MLQA benchmark..."
echo ""

python downstream_tasks/benchmarks/benchmark_mlqa.py \
    --model "$MODEL" \
    --train_epochs "$TRAIN_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --languages all \
    --full_matrix \
    --output_csv "$OUTPUT_CSV" \
    --output_dir "./results/mlqa_${MODEL//\//_}"

BENCHMARK_EXIT_CODE=$?

# --- Completion ---
echo ""
echo "=========================================="
echo "MLQA Benchmark Summary"
echo "=========================================="
echo "Exit Code: $BENCHMARK_EXIT_CODE"
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

if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ MLQA BENCHMARK COMPLETED SUCCESSFULLY"
else
    echo ""
    echo "❌ MLQA BENCHMARK FAILED"
fi

echo ""
echo "To compare with paper results, see Table 2 in:"
echo "https://arxiv.org/abs/1910.07475"
echo "=========================================="

exit $BENCHMARK_EXIT_CODE
