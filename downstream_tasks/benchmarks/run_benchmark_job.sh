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

# Simple BERT Benchmarking Job
# Runs xlm-roberta-base on 4 downstream tasks

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Navigate to project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"

# Activate virtual environment if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# GPU info
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Environment variables
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run benchmarks
cd downstream_tasks/benchmarks
bash run_all_benchmarks.sh xlm-roberta-base

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

