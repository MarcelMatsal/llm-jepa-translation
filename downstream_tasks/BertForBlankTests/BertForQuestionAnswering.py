"""
Question Answering task implementation for SQuAD and similar datasets.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, Optional, Tuple
import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoConfig,
    AutoModel,
)

from downstream_tasks.base_task import BaseTask
from downstream_tasks.task_configs import get_dataset_config, get_hyperparameters


def compute_qa_metrics(start_logits, end_logits, features, examples):
    """
    Compute exact match and F1 scores for question answering.
    This is a simplified version - in practice, you'd use the datasets library's metric.
    """
    # This is a placeholder - in practice, use datasets.load_metric("squad")
    # For now, return dummy metrics
    return {
        "exact_match": 0.0,
        "f1": 0.0,
    }


class QuestionAnsweringTask(BaseTask):
    """
    Question answering task (e.g., SQuAD).
    """
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        dataset_name: str = "squad",
        dataset_config: Optional[str] = "v1_1",
        output_dir: str = "./results",
        seed: int = 42,
        device: Optional[str] = None,
        max_length: int = 384,
        doc_stride: int = 128,
        **kwargs
    ):
        """
        Initialize question answering task.
        
        Args:
            model_name: HuggingFace model identifier
            dataset_name: Dataset name (e.g., 'squad')
            dataset_config: Dataset configuration (e.g., 'v1_1')
            output_dir: Output directory for results
            seed: Random seed
            device: Device to use
            max_length: Maximum sequence length
            doc_stride: Stride for long contexts
            **kwargs: Additional arguments
        """
        super().__init__(model_name, output_dir, seed, device)
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_info = get_dataset_config("question_answering", dataset_name, dataset_config)
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.context_column = self.dataset_info.get("text_column", "context")
        self.question_column = self.dataset_info.get("question_column", "question")
        self.answer_column = self.dataset_info.get("answer_column", "answers")
        self.squad_metric = None
    
    def prepare_model(self, base_model: Any, config: Any) -> Any:
        """
        Prepare model with question answering head.
        
        Args:
            base_model: Base encoder model
            config: Model configuration
            
        Returns:
            Model with QA head
        """
        # Create model with question answering head
        model = AutoModelForQuestionAnswering.from_pretrained(
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
        Load question answering dataset.
        
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
        
        # Load SQuAD metric
        try:
            from datasets import load_metric
            self.squad_metric = load_metric("squad")
        except Exception as e:
            print(f"  Warning: Could not load SQuAD metric: {e}")
            self.squad_metric = None
        
        print(f"  Context column: {self.context_column}")
        print(f"  Question column: {self.question_column}")
        print(f"  Answer column: {self.answer_column}")
        
        self.datasets = dataset
        print(f"  Train examples: {len(dataset['train'])}")
        if "validation" in dataset:
            print(f"  Validation examples: {len(dataset['validation'])}")
        if "test" in dataset:
            print(f"  Test examples: {len(dataset['test'])}")
        
        return dataset
    
    def preprocess_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess examples for question answering.
        Handles tokenization and answer position alignment.
        
        Args:
            examples: Batch of examples
            
        Returns:
            Tokenized examples with start/end positions
        """
        questions = examples[self.question_column]
        contexts = examples[self.context_column]
        answers = examples[self.answer_column]
        
        # Tokenize
        tokenized = self.tokenizer(
            questions,
            contexts,
            truncation="only_second",  # Only truncate context
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        # Extract offset mappings and map answers to token positions
        offset_mapping = tokenized.pop("offset_mapping")
        sample_map = tokenized.pop("overflow_to_sample_mapping")
        
        start_positions = []
        end_positions = []
        
        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            
            # Get character start/end positions
            if len(answer["answer_start"]) > 0:
                char_start = answer["answer_start"][0]
                char_end = char_start + len(answer["text"][0])
            else:
                char_start = 0
                char_end = 0
            
            # Find token positions
            sequence_ids = tokenized.sequence_ids(i)
            start_token = None
            end_token = None
            
            for token_idx, (start_char, end_char) in enumerate(offset):
                # Skip special tokens
                if sequence_ids[token_idx] != 1:  # 1 = context tokens
                    continue
                
                # Check if answer span is within this token
                if start_char <= char_start < end_char:
                    start_token = token_idx
                if start_char < char_end <= end_char:
                    end_token = token_idx
                    break
            
            # If answer not found, set to 0 (CLS token)
            if start_token is None or end_token is None:
                start_token = 0
                end_token = 0
            
            start_positions.append(start_token)
            end_positions.append(end_token)
        
        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        
        return tokenized
    
    def compute_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        Compute question answering metrics (exact match, F1).
        
        Args:
            eval_pred: Tuple of (predictions, labels)
            
        Returns:
            Dictionary of metrics
        """
        # For QA, we need to post-process predictions
        # This is a simplified version - in practice, use the datasets library
        predictions, labels = eval_pred
        
        start_logits, end_logits = predictions
        start_positions, end_positions = labels
        
        # Get predicted start/end positions
        pred_start = np.argmax(start_logits, axis=-1)
        pred_end = np.argmax(end_logits, axis=-1)
        
        # Compute simple accuracy on positions
        start_accuracy = (pred_start == start_positions).mean()
        end_accuracy = (pred_end == end_positions).mean()
        
        # For full EM/F1, we'd need to reconstruct answers and compare with ground truth
        # This requires the original examples, which we don't have here
        # In practice, use the datasets library's squad metric
        
        metrics = {
            "start_accuracy": float(start_accuracy),
            "end_accuracy": float(end_accuracy),
        }
        
        # If we have the squad metric, use it (requires post-processing in trainer)
        if self.squad_metric is not None:
            # Note: Full SQuAD evaluation requires post-processing predictions
            # This would typically be done in a custom trainer or post-processing step
            metrics["exact_match"] = 0.0  # Placeholder
            metrics["f1"] = 0.0  # Placeholder
        
        return metrics


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Question Answering Task")
    parser.add_argument("--model", type=str, default="xlm-roberta-base", help="Model name")
    parser.add_argument("--dataset", type=str, default="squad", help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="v1_1", help="Dataset config")
    parser.add_argument("--output_dir", type=str, default="./results/qa", help="Output directory")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for eval_only")
    
    args = parser.parse_args()
    
    # Create task
    task = QuestionAnsweringTask(
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
        hyperparams = get_hyperparameters("question_answering")
        task.train(**hyperparams)

