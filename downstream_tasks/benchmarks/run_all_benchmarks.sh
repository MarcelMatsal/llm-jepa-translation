#!/bin/bash
# Run all 4 benchmarks and consolidate results

cd "$(dirname "$0")"

MODEL="${1:-xlm-roberta-base}"
OUTPUT_CSV="benchmark_results.csv"

echo "Running all benchmarks with model: $MODEL"
echo ""

# Initialize CSV with header
echo "model,task,metric,value" > $OUTPUT_CSV

# Run all benchmarks
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
echo "All benchmarks completed!"
echo "=========================================="
echo ""
cat $OUTPUT_CSV
echo ""
echo "Results saved to $OUTPUT_CSV"

