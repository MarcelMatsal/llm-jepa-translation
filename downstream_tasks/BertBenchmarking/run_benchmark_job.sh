#!/bin/bash
#SBATCH --job-name=bert_benchmark
#SBATCH --output=logs/benchmark_output_%j.log
#SBATCH --error=logs/benchmark_error_%j.log
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# BERT Downstream Task Benchmarking
# Runs xlm-roberta-base on all 4 downstream tasks

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Navigate to project directory (adjust path if needed)
# Auto-detect: go up from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"

# Activate virtual environment (if exists)
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Verify GPU availability
echo ""
echo "=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi
echo ""

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run benchmarks
echo "=========================================="
echo "Running Benchmarks"
echo "=========================================="
echo ""

cd "$SCRIPT_DIR"
bash run_quick_benchmark.sh

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

