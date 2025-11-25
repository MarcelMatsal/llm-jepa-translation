#!/bin/bash
#SBATCH --job-name=validate_fix
#SBATCH --output=logs/validation_output_%j.log
#SBATCH --error=logs/validation_error_%j.log
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Navigate to project directory
cd /users/tgillin/files/llm-jepa-translation

# Activate virtual environment
source venv/bin/activate

# Run validation script
python tests/validate_fix.py
