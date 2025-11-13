#!/bin/bash

#SBATCH -p gpu                # Specify the 'gpu' partition
#SBATCH -N 1                  # Number of nodes
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH -n 1                  # Number of tasks
#SBATCH -c 8                  # Number of CPUs per task
#SBATCH --mem=64G             # Request 64GB of memory
#SBATCH -t 36:00:00           # Set a time limit of 36 hours
#SBATCH -J bert_benchmark     # Job name
#SBATCH -o slurm-%j.out       # Standard output file
#SBATCH -e slurm-%j.err       # Standard error file

# Simple BERT Benchmarking Job
# Runs benchmarks on 4 downstream tasks

# --- Environment Setup ---
echo "=========================================="
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="
echo ""

echo "Loading required modules..."
module load anaconda
module load cuda

# --- Job Execution ---
echo "Navigating to submission directory: $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

# GPU info
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Environment variables
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Get model name from argument (default: roberta-base)
MODEL="${1:-roberta-base}"
echo "Running benchmarks with model: $MODEL"
echo ""

# Run benchmarks
cd downstream_tasks/benchmarks

# NOTE: Update this path to match your conda environment
# Example: /users/YOUR_USERNAME/miniconda3/envs/YOUR_ENV/bin/python
PYTHON_PATH="/users/allalani/miniconda3/envs/csci1470/bin/python"

echo "Using Python: $PYTHON_PATH"
echo ""

$PYTHON_PATH -u benchmark_seq_class.py --model $MODEL
$PYTHON_PATH -u benchmark_ner.py --model $MODEL
$PYTHON_PATH -u benchmark_qa.py --model $MODEL
$PYTHON_PATH -u benchmark_mc.py --model $MODEL

echo ""
echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="

