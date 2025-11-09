#!/bin/bash
# Quick test script for downstream task evaluation
# Run this to verify your trained model has the right architecture

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
CHECKPOINT="experiments/exp_test/checkpoints/best_model"
DEVICE="cuda"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./downstream_tasks/run_quick_test.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH   Path to model checkpoint (default: experiments/exp_test/checkpoints/best_model)"
            echo "  --device DEVICE     Device to use: cuda or cpu (default: cuda)"
            echo "  --help              Show this help message"
            echo ""
            echo "Example:"
            echo "  ./downstream_tasks/run_quick_test.sh --checkpoint experiments/exp_test/checkpoints/checkpoint_epoch_5"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Downstream Task Quick Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Checkpoint: $CHECKPOINT"
echo "Device: $DEVICE"
echo ""

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT" ]; then
    echo -e "${YELLOW}⚠️  Warning: Checkpoint directory not found: $CHECKPOINT${NC}"
    echo "Please make sure the path is correct."
    exit 1
fi

# Test 1: Load model and verify architecture
echo -e "${GREEN}[1/3] Loading model and verifying architecture...${NC}"
echo "=================================================="
python downstream_tasks/load_model.py \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE"

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Model loading failed!${NC}"
    exit 1
fi

echo ""
read -p "Press Enter to continue to embedding tests..."
echo ""

# Test 2: Test embeddings
echo -e "${GREEN}[2/3] Testing sentence embeddings...${NC}"
echo "=================================================="
python downstream_tasks/test_embeddings.py \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE"

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Embedding test failed!${NC}"
    exit 1
fi

echo ""
read -p "Press Enter to continue to MLM tests..."
echo ""

# Test 3: Test MLM
echo -e "${GREEN}[3/3] Testing masked language modeling...${NC}"
echo "=================================================="
python downstream_tasks/test_mlm.py \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE"

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  MLM test failed!${NC}"
    exit 1
fi

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ ALL TESTS COMPLETED SUCCESSFULLY!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Your model has the correct BERT-compatible architecture and can be used for:"
echo "  • Sentence embeddings (CLS token extraction)"
echo "  • Cross-lingual similarity and retrieval"
echo "  • Masked language modeling"
echo "  • Fine-tuning on downstream tasks"
echo ""
echo "Next steps:"
echo "  1. Run interactive tests: python downstream_tasks/test_embeddings.py --checkpoint $CHECKPOINT --custom"
echo "  2. Fine-tune on your downstream task of choice"
echo "  3. Compare performance with baseline XLM-RoBERTa"
echo ""

