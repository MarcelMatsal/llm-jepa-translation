# BERT Downstream Task Benchmarking

Simple benchmarking system using HuggingFace's built-in infrastructure to evaluate BERT models on standard downstream tasks.

## Structure

```
benchmarks/
├── benchmark_seq_class.py    # SST-2 sentiment analysis
├── benchmark_ner.py          # CoNLL-2003 NER
├── benchmark_qa.py           # SQuAD v1.1 QA
├── benchmark_mc.py           # SWAG multiple choice
├── benchmark_mlqa.py         # MLQA cross-lingual QA (paper benchmark)
├── run_all_benchmarks.sh     # Run all tasks
├── run_mlqa_benchmark.sh     # SLURM script for MLQA
├── run_benchmark_job.sh      # SLURM submission script
└── README.md                 # This file
```

## Quick Start

### Run All Benchmarks

```bash
cd downstream_tasks/benchmarks
bash run_all_benchmarks.sh xlm-roberta-base
```

This will:
1. Run all 4 standard benchmarks sequentially
2. Save results to `benchmark_results.csv`
3. Display summary at the end

To also run MLQA (takes longer):
```bash
bash run_all_benchmarks.sh xlm-roberta-base true
```

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

# MLQA cross-lingual QA
python benchmark_mlqa.py --model xlm-roberta-base --languages all --cross_lingual
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

### MLQA (Cross-lingual QA)
- **Dataset**: MLQA (facebook/mlqa)
- **Paper**: [MLQA: Evaluating Cross-lingual Extractive Question Answering](https://arxiv.org/abs/1910.07475)
- **Metrics**: F1 Score, Exact Match (EM)
- **Languages**: English, Arabic, German, Spanish, Hindi, Vietnamese, Chinese
- **Methodology**: Train on SQuAD v1.1 (English), evaluate zero-shot on MLQA

## MLQA Benchmark

The MLQA benchmark implements the evaluation methodology from the original paper (Lewis et al., 2019).

### Usage

**Quick run (English only):**
```bash
python benchmark_mlqa.py --model xlm-roberta-base --languages en
```

**Full evaluation (all languages):**
```bash
python benchmark_mlqa.py \
    --model xlm-roberta-base \
    --languages all \
    --cross_lingual \
    --output_csv mlqa_results.csv
```

**Submit via SLURM:**
```bash
# Standard model
sbatch run_mlqa_benchmark.sh xlm-roberta-base

# Custom model on HuggingFace Hub
sbatch run_mlqa_benchmark.sh your-org/your-custom-model
```

### MLQA Arguments

- `--model`: HuggingFace model name or Hub repo
- `--train_epochs`: Training epochs on SQuAD (default: 2)
- `--languages`: Languages to evaluate: 'all' or comma-separated (e.g., 'en,de,es')
- `--cross_lingual`: Also evaluate cross-lingual pairs (English context, other-lang questions)
- `--skip_training`: Skip SQuAD training (use pre-trained QA model directly)
- `--output_csv`: Output CSV file

### MLQA Output Format

Results are saved to CSV with detailed language pair information:

```csv
model,task,context_lang,question_lang,f1,em,num_examples
xlm-roberta-base,mlqa_en_en,en,en,65.32,47.89,5495
xlm-roberta-base,mlqa_de_de,de,de,49.28,32.45,4517
xlm-roberta-base,mlqa_en_de,en,de,52.14,35.67,4517
...
```

### Comparing with Paper Results

The paper reports F1/EM scores for XLM-R base on MLQA (Table 2):

| Language | F1 | EM |
|----------|----|----|
| en | 77.4 | 64.6 |
| de | 60.6 | 44.5 |
| es | 66.5 | 47.7 |
| ar | 52.4 | 35.3 |
| hi | 58.5 | 43.3 |
| vi | 62.0 | 43.6 |
| zh | 56.4 | 37.3 |

Note: Results may vary slightly based on training configuration and random seed.

## Custom Models

Both standard HuggingFace models and custom models work the same way:

```bash
# Standard model
python benchmark_mlqa.py --model xlm-roberta-base

# Custom model on HuggingFace Hub
python benchmark_mlqa.py --model your-org/your-custom-model

# Local checkpoint
python benchmark_mlqa.py --model ./path/to/checkpoint
```

Custom models (like BertDualObjective) work because they save the base XLM-RoBERTa weights, which can be loaded by `AutoModelForQuestionAnswering`.

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
- `tqdm>=4.65.0`

## Notes

- All scripts use HuggingFace's `Trainer` API
- Results are saved to `results/<task>/` directories
- Each benchmark can be run independently
- CSV format makes it easy to compare models
- MLQA evaluation follows the paper's zero-shot transfer methodology
