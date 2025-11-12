"""
Token Classification task implementation for NER and other token-level tasks.
Supports CoNLL-2003 and similar datasets.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoConfig,
    AutoModel,
)
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import sys
import os
# Add BertBenchmarking directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_task import BaseTask
from task_configs import get_dataset_config, get_hyperparameters


class TokenClassificationTask(BaseTask):
    """
    Token classification task (e.g., Named Entity Recognition).
    """
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        dataset_name: str = "conll2003",
        dataset_config: Optional[str] = None,
        output_dir: str = "./results",
        seed: int = 42,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize token classification task.
        
        Args:
            model_name: HuggingFace model identifier
            dataset_name: Dataset name (e.g., 'conll2003')
            dataset_config: Dataset configuration
            output_dir: Output directory for results
            seed: Random seed
            device: Device to use
            **kwargs: Additional arguments
        """
        super().__init__(model_name, output_dir, seed, device)
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_info = get_dataset_config("token_classification", dataset_name, dataset_config)
        self.num_labels = None
        self.label_column = self.dataset_info.get("label_column", "ner_tags")
        self.text_column = self.dataset_info.get("text_column", "tokens")
        self.label_list = None
        self.label_to_id = None
        self.id_to_label = None
    
    def prepare_model(self, base_model: Any, config: Any) -> Any:
        """
        Prepare model with token classification head.
        
        Args:
            base_model: Base encoder model
            config: Model configuration
            
        Returns:
            Model with token classification head
        """
        # Determine number of labels (will be set after loading dataset)
        if self.num_labels is None:
            # Default to 9 for CoNLL-2003 (BIO scheme)
            num_labels = 9
        else:
            num_labels = self.num_labels
        
        # Create model with token classification head
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
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
        Load token classification dataset.
        
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
        
        # Determine number of labels and label mappings
        if "train" in dataset:
            # Get all unique labels
            all_labels = set()
            for example in dataset["train"]:
                labels = example[self.label_column]
                if isinstance(labels, list):
                    all_labels.update(labels)
            
            self.label_list = sorted(list(all_labels))
            self.num_labels = len(self.label_list)
            self.label_to_id = {label: i for i, label in enumerate(self.label_list)}
            self.id_to_label = {i: label for i, label in enumerate(self.label_list)}
            
            print(f"  Number of labels: {self.num_labels}")
            print(f"  Labels: {self.label_list}")
            print(f"  Label column: {self.label_column}")
            print(f"  Text column: {self.text_column}")
        
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
        Preprocess examples for token classification.
        
        Args:
            examples: Batch of examples
            
        Returns:
            Tokenized examples with aligned labels
        """
        # Get tokens and labels
        tokens = examples[self.text_column]
        labels = examples[self.label_column]
        
        # Tokenize
        tokenized = self.tokenizer(
            tokens,
            truncation=True,
            padding="max_length",
            max_length=128,
            is_split_into_words=True,  # Input is already tokenized
        )
        
        # Align labels with tokenizer output
        aligned_labels = []
        word_ids_list = []
        
        for i, label_seq in enumerate(labels):
            word_ids = tokenized.word_ids(batch_index=i)
            word_ids_list.append(word_ids)
            
            aligned_label_seq = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                # Special tokens get -100 (ignored in loss)
                if word_idx is None:
                    aligned_label_seq.append(-100)
                # Only label the first token of a word
                elif word_idx != previous_word_idx:
                    # Convert label to ID
                    if word_idx < len(label_seq):
                        label = label_seq[word_idx]
                        aligned_label_seq.append(self.label_to_id.get(label, 0))
                    else:
                        aligned_label_seq.append(-100)
                else:
                    # Subword token, use -100
                    aligned_label_seq.append(-100)
                
                previous_word_idx = word_idx
            
            aligned_labels.append(aligned_label_seq)
        
        tokenized["labels"] = aligned_labels
        
        return tokenized
    
    def compute_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        Compute token classification metrics (precision, recall, F1).
        Uses seqeval for proper NER evaluation.
        
        Args:
            eval_pred: Tuple of (predictions, labels)
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        
        # Get predicted labels
        predictions = np.argmax(predictions, axis=-1)
        
        # Convert to label sequences (ignoring -100)
        true_labels = []
        pred_labels = []
        
        for i in range(len(predictions)):
            true_seq = []
            pred_seq = []
            
            for j in range(len(predictions[i])):
                if labels[i][j] != -100:
                    true_seq.append(self.id_to_label[labels[i][j]])
                    pred_seq.append(self.id_to_label[predictions[i][j]])
            
            true_labels.append(true_seq)
            pred_labels.append(pred_seq)
        
        # Compute metrics using seqeval
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        
        return metrics


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Token Classification Task")
    parser.add_argument("--model", type=str, default="xlm-roberta-base", help="Model name")
    parser.add_argument("--dataset", type=str, default="conll2003", help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset config")
    parser.add_argument("--output_dir", type=str, default="./results/token_class", help="Output directory")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for eval_only")
    
    args = parser.parse_args()
    
    # Create task
    task = TokenClassificationTask(
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
        hyperparams = get_hyperparameters("token_classification")
        task.train(**hyperparams)

