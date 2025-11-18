#!/bin/bash
#SBATCH --job-name=cls_analysis
#SBATCH --output=logs/cls_analysis_output_%j.log
#SBATCH --error=logs/cls_analysis_error_%j.log
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# CLS Token Embedding Analysis Script
# Analyzes language embeddings from a multilingual model

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

# Parse command line arguments with defaults
MODEL_NAME="${1:-xlm-roberta-base}"
NUM_LANGUAGES="${2:-10}"
SAMPLES_PER_LANG="${3:-100}"
OUTPUT_DIR="${4:-analysis_output/$(date +%Y%m%d_%H%M%S)}"
DEVICE="${5:-cuda}"

echo "=========================================="
echo "Analysis Configuration"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Number of languages: $NUM_LANGUAGES"
echo "Samples per language: $SAMPLES_PER_LANG"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo ""

# Install required packages if not already installed
echo "Checking dependencies..."
pip install scikit-learn>=1.0.0 --quiet
pip install seaborn>=0.11.0 --quiet
pip install matplotlib>=3.4.0 --quiet
echo "✓ Dependencies verified"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# =========================================
# Run CLS Token Embedding Analysis
# =========================================
echo ""
echo "=========================================="
echo "Running CLS Token Embedding Analysis"
echo "=========================================="
echo ""

python scripts/analyze_cls_embeddings.py \
    --model-name "$MODEL_NAME" \
    --num-languages "$NUM_LANGUAGES" \
    --samples-per-language "$SAMPLES_PER_LANG" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --batch-size 32 \
    --seed 42 \
    --min-length 20 \
    --max-length 200

EXIT_CODE=$?

# =========================================
# Summary
# =========================================
echo ""
echo "=========================================="
echo "Analysis Summary"
echo "=========================================="
echo "End Time: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ANALYSIS COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "  Visualizations:"
    echo "    • pca_2d.png       - PCA 2D projection"
    echo "    • pca_3d.png       - PCA 3D projection"
    echo "    • tsne_2d.png      - t-SNE 2D projection"
    echo "    • tsne_3d.png      - t-SNE 3D projection"
    echo "    • eigenspectrum.png - Eigenvalue spectrum"
    echo ""
    echo "  Data files:"
    echo "    • embeddings.npz         - Raw embeddings and projections"
    echo "    • metadata.json          - Analysis metadata"
    echo "    • analysis_summary.txt   - Detailed summary"
    echo ""
    echo "To view results:"
    echo "  cd $OUTPUT_DIR"
    echo "  ls -lh *.png"
    echo ""
else
    echo "❌ ANALYSIS FAILED"
    echo "Exit code: $EXIT_CODE"
    echo ""
    echo "Check the log files for details:"
    echo "  Output: logs/cls_analysis_output_$SLURM_JOB_ID.log"
    echo "  Error:  logs/cls_analysis_error_$SLURM_JOB_ID.log"
fi

echo "=========================================="

exit $EXIT_CODE

