# BERT Downstream Task Benchmarking

Simple benchmarking system using HuggingFace's built-in infrastructure to evaluate BERT models on standard downstream tasks.

## Structure

```
benchmarks/
├── benchmark_seq_class.py    # SST-2 sentiment analysis
├── benchmark_ner.py           # CoNLL-2003 NER
├── benchmark_qa.py            # SQuAD v1.1 QA
├── benchmark_mc.py            # SWAG multiple choice
├── run_all_benchmarks.sh      # Run all 4 tasks
├── run_benchmark_job.sh       # SLURM submission script
└── README.md                  # This file
```

## Quick Start

### Run All Benchmarks

```bash
cd downstream_tasks/benchmarks
bash run_all_benchmarks.sh xlm-roberta-base
```

This will:
1. Run all 4 benchmarks sequentially
2. Save results to `benchmark_results.csv`
3. Display summary at the end

### Submit to SLURM

```bash
sbatch downstream_tasks/benchmarks/run_benchmark_job.sh
```

Check job status:
```bash
squeue -u $USER
tail -f logs/benchmark_output_<job_id>.log
```

### Run Individual Benchmarks

```bash
# Sequence classification (SST-2 - default)
python benchmark_seq_class.py --model xlm-roberta-base

# NER (CoNLL-2003 - default)
python benchmark_ner.py --model xlm-roberta-base

# QA (SQuAD - default)
python benchmark_qa.py --model xlm-roberta-base

# Multiple choice (SWAG - default)
python benchmark_mc.py --model xlm-roberta-base
```

### Customize Dataset and Hyperparameters

All scripts support these arguments:

```bash
# Use different model
python benchmark_seq_class.py --model bert-base-uncased

# Use different dataset
python benchmark_seq_class.py --dataset glue --dataset_config mrpc

# Customize hyperparameters
python benchmark_ner.py --model roberta-base --epochs 5 --batch_size 32 --learning_rate 3e-5

# Mix and match
python benchmark_qa.py \
    --model xlm-roberta-large \
    --dataset squad_v2 \
    --epochs 3 \
    --batch_size 16
```

**Available arguments:**
- `--model`: HuggingFace model name (default: xlm-roberta-base)
- `--dataset`: Dataset name (default: varies by task)
- `--dataset_config`: Dataset config (default: varies by task)
- `--epochs`: Number of training epochs (default: 3 for most, 2 for QA)
- `--batch_size`: Training batch size (default: 16 for most, 12 for QA, 8 for MC)
- `--learning_rate`: Learning rate (default: 2e-5 for most, 3e-5 for QA)
- `--seed`: Random seed (default: 42)
- `--output_csv`: Output CSV file (default: benchmark_results.csv)

## Output

Results are saved to `benchmark_results.csv`:

```csv
model,task,metric,value
xlm-roberta-base,sst2,accuracy,0.9234
xlm-roberta-base,conll2003,f1,0.8945
xlm-roberta-base,squad,completed,1.0
xlm-roberta-base,swag,accuracy,0.7812
```

## Compare Models

To compare your finetuned model against the baseline:

```bash
# Run baseline
bash run_all_benchmarks.sh xlm-roberta-base

# Run your model
bash run_all_benchmarks.sh your-org/your-model

# Results will be appended to the same CSV
cat benchmark_results.csv
```

## Task Details

### SST-2 (Sequence Classification)
- **Dataset**: GLUE SST-2 (binary sentiment)
- **Metric**: Accuracy
- **Epochs**: 3
- **Batch size**: 16

### CoNLL-2003 (NER)
- **Dataset**: CoNLL-2003 (named entity recognition)
- **Metric**: F1 score
- **Epochs**: 3
- **Batch size**: 16

### SQuAD v1.1 (Question Answering)
- **Dataset**: SQuAD v1.1
- **Metric**: Training completion marker
- **Epochs**: 2
- **Batch size**: 12
- **Note**: Full EM/F1 evaluation requires post-processing

### SWAG (Multiple Choice)
- **Dataset**: SWAG (commonsense reasoning)
- **Metric**: Accuracy
- **Epochs**: 3
- **Batch size**: 8

## Requirements

Make sure you have installed:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `transformers>=4.35.0`
- `datasets>=2.14.0`
- `evaluate>=0.4.0`
- `seqeval>=1.2.2`
- `pandas>=1.3.0`

## Notes

- All scripts use HuggingFace's `Trainer` API
- Results are saved to `results/<task>/` directories
- Each benchmark can be run independently
- CSV format makes it easy to compare models

