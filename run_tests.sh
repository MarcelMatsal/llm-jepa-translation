#!/bin/bash
#SBATCH --job-name=llm_jepa_tests
#SBATCH --output=logs/test_output_%j.log
#SBATCH --error=logs/test_error_%j.log
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=batch

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

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "RUNNING TESTS"
echo "=========================================="
echo ""

# Run data pipeline tests
echo "=========================================="
echo "Running Data Pipeline Tests..."
echo "=========================================="
python tests/test_data.py
DATA_EXIT_CODE=$?

echo ""

# Run model tests
echo "=========================================="
echo "Running Model Tests..."
echo "=========================================="
python tests/test_model.py
MODEL_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo "Data Tests Exit Code: $DATA_EXIT_CODE"
echo "Model Tests Exit Code: $MODEL_EXIT_CODE"

if [ $DATA_EXIT_CODE -eq 0 ] && [ $MODEL_EXIT_CODE -eq 0 ]; then
    echo "✓ ALL TESTS PASSED"
    echo "=========================================="
    exit 0
else
    echo "❌ SOME TESTS FAILED"
    echo "=========================================="
    exit 1
fi

echo ""
echo "End Time: $(date)"

