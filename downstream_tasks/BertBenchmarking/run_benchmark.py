"""
Main script to train and benchmark BERT models on downstream tasks.
Supports sequence classification, token classification, question answering, and multiple choice.
"""
import sys
import os

# Add BertBenchmarking directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from typing import Optional

from BertForBlankTests.BertForSequenceClassification import SequenceClassificationTask
from BertForBlankTests.BertForTokenClassification import TokenClassificationTask
from BertForBlankTests.BertForQuestionAnswering import QuestionAnsweringTask
from BertForBlankTests.BertForMultipleChoice import MultipleChoiceTask
from task_configs import get_hyperparameters, list_available_datasets


def main():
    parser = argparse.ArgumentParser(
        description="Train and benchmark BERT models on downstream tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on SST-2 (sequence classification)
  python run_benchmark.py --task sequence_classification --dataset glue --dataset_config sst2 --model xlm-roberta-base

  # Train on CoNLL-2003 (token classification)
  python run_benchmark.py --task token_classification --dataset conll2003 --model xlm-roberta-base

  # Train on SQuAD (question answering)
  python run_benchmark.py --task question_answering --dataset squad --dataset_config v1_1 --model xlm-roberta-base

  # Train on SWAG (multiple choice)
  python run_benchmark.py --task multiple_choice --dataset swag --model xlm-roberta-base

  # Evaluate only (no training)
  python run_benchmark.py --task sequence_classification --dataset glue --dataset_config sst2 --eval_only --checkpoint ./results/sst2_xlm_roberta

  # Compare with your model
  python run_benchmark.py --task sequence_classification --dataset glue --dataset_config sst2 --model User/your-model --output_dir results/sst2_your_model
        """
    )
    
    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["sequence_classification", "token_classification", "question_answering", "multiple_choice"],
        help="Type of task to run"
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="xlm-roberta-base",
        help="HuggingFace model identifier (default: xlm-roberta-base)"
    )
    
    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'glue', 'conll2003', 'squad', 'swag')"
    )
    
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Dataset configuration (e.g., 'sst2', 'v1_1')"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/{task}_{dataset}_{model_name})"
    )
    
    # Training options
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only evaluate, don't train (requires --checkpoint)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for evaluation (only used with --eval_only)"
    )
    
    # Hyperparameters (optional overrides)
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides default)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides default)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size (overrides default)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    parser.add_argument(
        "--list_datasets",
        action="store_true",
        help="List available datasets for the specified task and exit"
    )
    
    args = parser.parse_args()
    
    # List datasets if requested
    if args.list_datasets:
        print(f"\nAvailable datasets for task '{args.task}':")
        try:
            datasets = list_available_datasets(args.task)
            for dataset_name, configs in datasets.items():
                print(f"  {dataset_name}:")
                for config in configs:
                    print(f"    - {config}")
        except ValueError as e:
            print(f"Error: {e}")
        return
    
    # Determine output directory
    if args.output_dir is None:
        model_name_short = args.model.split("/")[-1].replace("-", "_")
        dataset_short = args.dataset
        if args.dataset_config:
            dataset_short += f"_{args.dataset_config}"
        args.output_dir = f"results/{args.task}_{dataset_short}_{model_name_short}"
    
    print("="*80)
    print("BERT Downstream Task Benchmarking")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}" + (f" ({args.dataset_config})" if args.dataset_config else ""))
    print(f"Output: {args.output_dir}")
    print("="*80)
    
    # Get hyperparameters
    hyperparams = get_hyperparameters(args.task)
    
    # Override with command-line arguments
    if args.num_epochs is not None:
        hyperparams["num_epochs"] = args.num_epochs
    if args.learning_rate is not None:
        hyperparams["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        hyperparams["batch_size"] = args.batch_size
    
    # Create task instance
    task_kwargs = {
        "model_name": args.model,
        "dataset_name": args.dataset,
        "dataset_config": args.dataset_config,
        "output_dir": args.output_dir,
        "seed": args.seed,
    }
    
    if args.task == "sequence_classification":
        task = SequenceClassificationTask(**task_kwargs)
    elif args.task == "token_classification":
        task = TokenClassificationTask(**task_kwargs)
    elif args.task == "question_answering":
        task = QuestionAnsweringTask(**task_kwargs)
    elif args.task == "multiple_choice":
        task = MultipleChoiceTask(**task_kwargs)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    # Load dataset
    task.load_dataset()
    
    # Train or evaluate
    if args.eval_only:
        if args.checkpoint is None:
            print("Error: --checkpoint required when using --eval_only")
            return
        task.evaluate_only(args.checkpoint)
    else:
        task.train(**hyperparams)
    
    print("\n" + "="*80)
    print("Benchmark Complete!")
    print("="*80)
    print(f"Results saved to: {args.output_dir}/results.json")


if __name__ == "__main__":
    main()

