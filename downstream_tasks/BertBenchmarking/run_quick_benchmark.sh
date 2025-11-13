#!/bin/bash
# Run xlm-roberta-base on all 4 downstream tasks with default configs

cd "$(dirname "$0")"

MODEL="xlm-roberta-base"

echo "Running baseline benchmarks with xlm-roberta-base..."
echo ""

# Sequence Classification - SST-2
python run_benchmark.py --task sequence_classification --dataset glue --dataset_config sst2 --model $MODEL

# Token Classification - CoNLL-2003
python run_benchmark.py --task token_classification --dataset conll2003 --model $MODEL

# Question Answering - SQuAD v1.1
python run_benchmark.py --task question_answering --dataset squad --dataset_config v1_1 --model $MODEL

# Multiple Choice - SWAG
python run_benchmark.py --task multiple_choice --dataset swag --model $MODEL

echo ""
echo "Done! Results saved in results/ directories."
