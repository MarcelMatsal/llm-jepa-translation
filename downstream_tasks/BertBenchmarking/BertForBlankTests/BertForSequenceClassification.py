"""
Sequence Classification task implementation for GLUE and other classification datasets.
Supports single-sentence and sentence-pair classification.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, Optional, Tuple
import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoModel,
)
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

import sys
import os
# Add BertBenchmarking directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_task import BaseTask
from task_configs import get_dataset_config, get_hyperparameters


class SequenceClassificationTask(BaseTask):
    """
    Sequence classification task (e.g., sentiment analysis, NLI).
    Supports both single-sentence and sentence-pair tasks.
    """
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        dataset_name: str = "glue",
        dataset_config: Optional[str] = "sst2",
        output_dir: str = "./results",
        seed: int = 42,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize sequence classification task.
        
        Args:
            model_name: HuggingFace model identifier
            dataset_name: Dataset name (e.g., 'glue')
            dataset_config: Dataset configuration (e.g., 'sst2', 'mrpc')
            output_dir: Output directory for results
            seed: Random seed
            device: Device to use
            **kwargs: Additional arguments
        """
        super().__init__(model_name, output_dir, seed, device)
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_info = get_dataset_config("sequence_classification", dataset_name, dataset_config)
        self.num_labels = None
        self.label_column = self.dataset_info.get("label_column", "label")
        self.text_column = self.dataset_info.get("text_column", "sentence")
        self.text2_column = self.dataset_info.get("text2_column")  # For sentence-pair tasks
    
    def prepare_model(self, base_model: Any, config: Any) -> Any:
        """
        Prepare model with sequence classification head.
        
        Args:
            base_model: Base encoder model (not used directly, but kept for interface)
            config: Model configuration
            
        Returns:
            Model with classification head
        """
        # Determine number of labels (will be set after loading dataset)
        if self.num_labels is None:
            # Default to 2 for binary classification
            num_labels = 2
        else:
            num_labels = self.num_labels
        
        # Create model with classification head
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True  # In case base model has different head
        )
        
        return model
    
    def load_dataset(
        self,
        dataset_name: Optional[str] = None,
        dataset_config: Optional[str] = None,
        cache_dir: Optional[str] = None
    ) -> DatasetDict:
        """
        Load sequence classification dataset.
        
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
        
        # Determine number of labels
        if "train" in dataset:
            label_list = dataset["train"].unique(self.label_column)
            self.num_labels = len(label_list)
            print(f"  Number of labels: {self.num_labels}")
            print(f"  Label column: {self.label_column}")
            print(f"  Text column: {self.text_column}")
            if self.text2_column:
                print(f"  Text2 column: {self.text2_column}")
        
        # If model was already loaded, reload it with correct number of labels
        if self.model is not None:
            print("  Reloading model with correct number of labels...")
            self.load_model_and_tokenizer()
        
        self.datasets = dataset
        print(f"  Train examples: {len(dataset['train'])}")
        if "validation" in dataset:
            print(f"  Validation examples: {len(dataset['validation'])}")
        if "test" in dataset:
            print(f"  Test examples: {len(dataset['test'])}")
        
        return dataset
    
    def preprocess_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess examples for sequence classification.
        Handles both single-sentence and sentence-pair tasks.
        
        Args:
            examples: Batch of examples
            
        Returns:
            Tokenized examples
        """
        # Get text columns
        texts = examples[self.text_column]
        
        # Handle sentence-pair tasks
        if self.text2_column and self.text2_column in examples:
            texts2 = examples[self.text2_column]
            # Tokenize pairs
            result = self.tokenizer(
                texts,
                texts2,
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        else:
            # Tokenize single sentences
            result = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        
        # Add labels
        if self.label_column in examples:
            result["labels"] = examples[self.label_column]
        
        return result
    
    def compute_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        Compute classification metrics (accuracy, F1, Matthews correlation).
        
        Args:
            eval_pred: Tuple of (predictions, labels)
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        
        # Get predicted labels
        predictions = np.argmax(predictions, axis=1)
        
        # Compute metrics
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        
        metrics = {
            "accuracy": accuracy,
            "f1": f1,
        }
        
        # Add Matthews correlation for binary classification
        if self.num_labels == 2:
            mcc = matthews_corrcoef(labels, predictions)
            metrics["matthews_correlation"] = mcc
        
        return metrics


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Sequence Classification Task")
    parser.add_argument("--model", type=str, default="xlm-roberta-base", help="Model name")
    parser.add_argument("--dataset", type=str, default="glue", help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="sst2", help="Dataset config")
    parser.add_argument("--output_dir", type=str, default="./results/seq_class", help="Output directory")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for eval_only")
    
    args = parser.parse_args()
    
    # Create task
    task = SequenceClassificationTask(
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
        hyperparams = get_hyperparameters("sequence_classification")
        task.train(**hyperparams)

