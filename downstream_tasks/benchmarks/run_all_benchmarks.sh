#!/bin/bash
# Run all 5 benchmarks and consolidate results

cd "$(dirname "$0")"

MODEL="${1:-xlm-roberta-base}"
OUTPUT_CSV="benchmark_results.csv"
MLQA_CSV="mlqa_results.csv"
RUN_MLQA="${2:-false}"  # Set to 'true' to include MLQA (takes longer)

echo "Running benchmarks with model: $MODEL"
echo ""

# Initialize CSV with header
echo "model,task,metric,value" > $OUTPUT_CSV

# Run standard benchmarks
echo "1/4 Running SST-2 (Sequence Classification)..."
python benchmark_seq_class.py --model $MODEL --output_csv $OUTPUT_CSV

echo ""
echo "2/4 Running CoNLL-2003 (NER)..."
python benchmark_ner.py --model $MODEL --output_csv $OUTPUT_CSV

echo ""
echo "3/4 Running SQuAD (QA)..."
python benchmark_qa.py --model $MODEL --output_csv $OUTPUT_CSV

echo ""
echo "4/4 Running SWAG (Multiple Choice)..."
python benchmark_mc.py --model $MODEL --output_csv $OUTPUT_CSV

echo ""
echo "=========================================="
echo "Standard benchmarks completed!"
echo "=========================================="
echo ""
cat $OUTPUT_CSV
echo ""
echo "Results saved to $OUTPUT_CSV"

# Optionally run MLQA (cross-lingual QA)
if [ "$RUN_MLQA" = "true" ]; then
    echo ""
    echo "=========================================="
    echo "5/5 Running MLQA (Cross-lingual QA)..."
    echo "=========================================="
    echo "Note: MLQA takes longer as it trains on SQuAD and evaluates on 7 languages"
    echo ""
    python benchmark_mlqa.py \
        --model $MODEL \
        --languages all \
        --cross_lingual \
        --output_csv $MLQA_CSV
    
    echo ""
    echo "MLQA results saved to $MLQA_CSV"
fi

echo ""
echo "=========================================="
echo "All benchmarks completed!"
echo "=========================================="

