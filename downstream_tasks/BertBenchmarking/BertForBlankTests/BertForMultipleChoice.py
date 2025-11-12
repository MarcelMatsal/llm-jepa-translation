"""
Multiple Choice task implementation for SWAG, RACE, and similar datasets.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForMultipleChoice,
    AutoConfig,
    AutoModel,
)

import sys
import os
# Add BertBenchmarking directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_task import BaseTask
from task_configs import get_dataset_config, get_hyperparameters


class MultipleChoiceTask(BaseTask):
    """
    Multiple choice task (e.g., SWAG, RACE).
    """
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        dataset_name: str = "swag",
        dataset_config: Optional[str] = None,
        output_dir: str = "./results",
        seed: int = 42,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize multiple choice task.
        
        Args:
            model_name: HuggingFace model identifier
            dataset_name: Dataset name (e.g., 'swag', 'race')
            dataset_config: Dataset configuration
            output_dir: Output directory for results
            seed: Random seed
            device: Device to use
            **kwargs: Additional arguments
        """
        super().__init__(model_name, output_dir, seed, device)
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_info = get_dataset_config("multiple_choice", dataset_name, dataset_config)
        self.context_column = self.dataset_info.get("context_column", "sent1")
        self.ending_column = self.dataset_info.get("ending_column", "ending")
        self.question_column = self.dataset_info.get("question_column")
        self.options_column = self.dataset_info.get("options_column")
        self.label_column = self.dataset_info.get("label_column", "label")
        self.num_choices = 4  # Default for SWAG, may vary for other datasets
    
    def prepare_model(self, base_model: Any, config: Any) -> Any:
        """
        Prepare model with multiple choice head.
        
        Args:
            base_model: Base encoder model
            config: Model configuration
            
        Returns:
            Model with multiple choice head
        """
        # Create model with multiple choice head
        model = AutoModelForMultipleChoice.from_pretrained(
            self.model_name,
            ignore_mismatched_sizes=True
        )
        
        return model
    
    def load_dataset(
        self,
        dataset_name: Optional[str] = None,
        dataset_config: Optional[str] = None,
        cache_dir: Optional[str] = None
    ) -> DatasetDict:
        """
        Load multiple choice dataset.
        
        Args:
            dataset_name: Dataset name (uses self.dataset_name if None)
            dataset_config: Dataset config (uses self.dataset_config if None)
            cache_dir: Cache directory
            
        Returns:
            DatasetDict with train/validation/test splits
        """
        if dataset_name is None:
            dataset_name = self.dataset_info["name"]
        if dataset_config is None:
            dataset_config = self.dataset_info.get("config")
        
        print(f"Loading dataset: {dataset_name}" + (f" ({dataset_config})" if dataset_config else ""))
        
        # Load dataset
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, cache_dir=cache_dir)
        else:
            dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        
        # Determine number of choices
        if "train" in dataset:
            # For SWAG, endings are in a list
            if self.ending_column and self.ending_column in dataset["train"][0]:
                endings = dataset["train"][0][self.ending_column]
                if isinstance(endings, list):
                    self.num_choices = len(endings)
            # For RACE, options are in a list
            elif self.options_column and self.options_column in dataset["train"][0]:
                options = dataset["train"][0][self.options_column]
                if isinstance(options, list):
                    self.num_choices = len(options)
            
            print(f"  Number of choices: {self.num_choices}")
            print(f"  Context column: {self.context_column}")
            if self.ending_column:
                print(f"  Ending column: {self.ending_column}")
            if self.question_column:
                print(f"  Question column: {self.question_column}")
            if self.options_column:
                print(f"  Options column: {self.options_column}")
            print(f"  Label column: {self.label_column}")
        
        self.datasets = dataset
        print(f"  Train examples: {len(dataset['train'])}")
        if "validation" in dataset:
            print(f"  Validation examples: {len(dataset['validation'])}")
        if "test" in dataset:
            print(f"  Test examples: {len(dataset['test'])}")
        
        return dataset
    
    def preprocess_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess examples for multiple choice.
        Creates separate inputs for each choice.
        
        Args:
            examples: Batch of examples
            
        Returns:
            Tokenized examples with flattened choices
        """
        # Get context
        contexts = examples[self.context_column]
        
        # Get choices (endings or options)
        if self.ending_column and self.ending_column in examples:
            # SWAG format: list of endings
            choices_list = examples[self.ending_column]
        elif self.options_column and self.options_column in examples:
            # RACE format: list of options
            choices_list = examples[self.options_column]
        else:
            raise ValueError("No ending_column or options_column found")
        
        # Get questions if available (for RACE)
        questions = None
        if self.question_column and self.question_column in examples:
            questions = examples[self.question_column]
        
        # Flatten: create one input per choice
        first_sentences = []
        second_sentences = []
        
        for i, context in enumerate(contexts):
            choices = choices_list[i]
            
            for choice in choices:
                first_sentences.append(context)
                
                # Format: context + question + choice (for RACE) or context + choice (for SWAG)
                if questions is not None:
                    question = questions[i]
                    second_sentences.append(f"{question} {choice}")
                else:
                    second_sentences.append(choice)
        
        # Tokenize all pairs
        tokenized = self.tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        
        # Reshape: (batch_size * num_choices, seq_len) -> (batch_size, num_choices, seq_len)
        batch_size = len(contexts)
        num_choices = self.num_choices
        
        tokenized = {
            k: [v[i:i + num_choices] for i in range(0, len(v), num_choices)]
            for k, v in tokenized.items()
        }
        
        # Add labels
        if self.label_column in examples:
            tokenized["labels"] = examples[self.label_column]
        
        return tokenized
    
    def compute_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        Compute multiple choice metrics (accuracy).
        
        Args:
            eval_pred: Tuple of (predictions, labels)
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        
        # Get predicted choices
        predictions = np.argmax(predictions, axis=-1)
        
        # Compute accuracy
        accuracy = (predictions == labels).mean()
        
        metrics = {
            "accuracy": float(accuracy),
        }
        
        return metrics


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Multiple Choice Task")
    parser.add_argument("--model", type=str, default="xlm-roberta-base", help="Model name")
    parser.add_argument("--dataset", type=str, default="swag", help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset config")
    parser.add_argument("--output_dir", type=str, default="./results/multiple_choice", help="Output directory")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for eval_only")
    
    args = parser.parse_args()
    
    # Create task
    task = MultipleChoiceTask(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
    )
    
    # Load dataset
    task.load_dataset()
    
    # Train or evaluate
    if args.eval_only:
        task.evaluate_only(args.checkpoint)
    else:
        # Get hyperparameters
        hyperparams = get_hyperparameters("multiple_choice")
        task.train(**hyperparams)

