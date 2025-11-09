#!/bin/bash
#SBATCH --job-name=downstream_tests
#SBATCH --output=logs/downstream_output_%j.log
#SBATCH --error=logs/downstream_error_%j.log
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Downstream Task Testing Script
# Tests that the trained model has correct BERT-compatible architecture

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

# Parse command line arguments
CHECKPOINT="${1:-experiments/exp_test/checkpoints_small/best_model}"
DEVICE="${2:-cuda}"

echo "=========================================="
echo "Test Configuration"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Device: $DEVICE"
echo ""

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT" ]; then
    echo "❌ ERROR: Checkpoint directory not found: $CHECKPOINT"
    echo ""
    echo "Available checkpoints:"
    ls -la experiments/exp_test/checkpoints_small/ 2>/dev/null || echo "  No checkpoints found"
    exit 1
fi

# Install scipy if not already installed (needed for embedding tests)
echo "Checking dependencies..."
pip install scipy>=1.9.0 --quiet
echo "✓ Dependencies verified"
echo ""

# =========================================
# TEST 1: Load Model and Verify Architecture
# =========================================
echo ""
echo "=========================================="
echo "TEST 1: Architecture Verification"
echo "=========================================="
python downstream_tasks/load_model.py \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE"

TEST1_EXIT_CODE=$?

if [ $TEST1_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ Architecture verification FAILED"
    echo "Exit code: $TEST1_EXIT_CODE"
    exit 1
fi

echo ""
echo "✓ Architecture verification PASSED"

# =========================================
# TEST 2: Sentence Embeddings
# =========================================
echo ""
echo "=========================================="
echo "TEST 2: Sentence Embedding Extraction"
echo "=========================================="
python downstream_tasks/test_embeddings.py \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE"

TEST2_EXIT_CODE=$?

if [ $TEST2_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ Embedding test FAILED"
    echo "Exit code: $TEST2_EXIT_CODE"
    exit 1
fi

echo ""
echo "✓ Embedding test PASSED"

# =========================================
# TEST 3: Masked Language Modeling
# =========================================
echo ""
echo "=========================================="
echo "TEST 3: Masked Language Modeling"
echo "=========================================="
python downstream_tasks/test_mlm.py \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE"

TEST3_EXIT_CODE=$?

if [ $TEST3_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ MLM test FAILED"
    echo "Exit code: $TEST3_EXIT_CODE"
    exit 1
fi

echo ""
echo "✓ MLM test PASSED"

# =========================================
# Summary
# =========================================
echo ""
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo "Test 1 (Architecture):  ✓ PASSED"
echo "Test 2 (Embeddings):    ✓ PASSED"
echo "Test 3 (MLM):          ✓ PASSED"
echo ""
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "✅ ALL DOWNSTREAM TESTS PASSED!"
echo ""
echo "Your model has the correct BERT-compatible architecture."
echo "It can be used for:"
echo "  • Sentence embeddings (CLS token extraction)"
echo "  • Cross-lingual similarity and retrieval"
echo "  • Masked language modeling"
echo "  • Fine-tuning on downstream tasks"
echo ""
echo "Logs saved to:"
echo "  Output: logs/downstream_output_$SLURM_JOB_ID.log"
echo "  Error:  logs/downstream_error_$SLURM_JOB_ID.log"
echo "=========================================="

exit 0

