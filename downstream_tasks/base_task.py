"""
Base task interface for downstream BERT benchmarking.
All task implementations inherit from this class.
"""
import os
import json
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    set_seed
)
from datasets import DatasetDict, Dataset
import numpy as np


class BaseTask(ABC):
    """
    Abstract base class for all downstream tasks.
    Provides common functionality for model loading, dataset preparation, and training.
    """
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        output_dir: str = "./results",
        seed: int = 42,
        device: Optional[str] = None
    ):
        """
        Initialize the task.
        
        Args:
            model_name: HuggingFace model identifier
            output_dir: Directory to save results
            seed: Random seed for reproducibility
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.seed = seed
        set_seed(seed)
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Will be set by subclasses
        self.tokenizer = None
        self.model = None
        self.datasets = None
        self.metrics = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """
        Load model and tokenizer from HuggingFace.
        Subclasses should override prepare_model() to add task-specific heads.
        
        Returns:
            model, tokenizer
        """
        print(f"Loading model: {self.model_name}")
        print(f"Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load base model config to determine architecture
        config = AutoConfig.from_pretrained(self.model_name)
        
        # Load base encoder (will be wrapped by task-specific model)
        base_model = AutoModel.from_pretrained(self.model_name)
        
        # Prepare task-specific model (implemented by subclasses)
        self.model = self.prepare_model(base_model, config)
        self.model.to(self.device)
        
        print(f"✓ Model loaded: {type(self.model).__name__}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model, self.tokenizer
    
    @abstractmethod
    def prepare_model(self, base_model: Any, config: Any) -> Any:
        """
        Prepare task-specific model by adding appropriate head to base encoder.
        
        Args:
            base_model: Base encoder model
            config: Model configuration
            
        Returns:
            Task-specific model with head
        """
        pass
    
    @abstractmethod
    def load_dataset(
        self,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        cache_dir: Optional[str] = None
    ) -> DatasetDict:
        """
        Load and preprocess dataset for the task.
        
        Args:
            dataset_name: HuggingFace dataset name
            dataset_config: Optional dataset configuration
            cache_dir: Optional cache directory
            
        Returns:
            DatasetDict with 'train', 'validation', and optionally 'test' splits
        """
        pass
    
    @abstractmethod
    def preprocess_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess examples for the task (tokenization, formatting).
        
        Args:
            examples: Batch of examples from dataset
            
        Returns:
            Preprocessed examples
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        Compute task-specific metrics.
        
        Args:
            eval_pred: Tuple of (predictions, labels)
            
        Returns:
            Dictionary of metric names to values
        """
        pass
    
    def get_training_arguments(
        self,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        eval_batch_size: Optional[int] = None,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        max_length: int = 512,
        save_strategy: str = "epoch",
        evaluation_strategy: str = "epoch",
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False,
        **kwargs
    ) -> TrainingArguments:
        """
        Get training arguments for HuggingFace Trainer.
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Training batch size
            eval_batch_size: Evaluation batch size (defaults to batch_size)
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay
            max_length: Maximum sequence length
            save_strategy: When to save checkpoints
            evaluation_strategy: When to evaluate
            load_best_model_at_end: Whether to load best model at end
            metric_for_best_model: Metric to use for best model selection
            greater_is_better: Whether higher metric is better
            **kwargs: Additional arguments to pass to TrainingArguments
            
        Returns:
            TrainingArguments object
        """
        if eval_batch_size is None:
            eval_batch_size = batch_size
        
        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=100,
            save_strategy=save_strategy,
            evaluation_strategy=evaluation_strategy,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            seed=self.seed,
            data_seed=self.seed,
            fp16=self.device == "cuda",
            report_to="none",  # Disable wandb/tensorboard by default
            **kwargs
        )
        
        return args
    
    def train(
        self,
        training_args: Optional[TrainingArguments] = None,
        **training_kwargs
    ) -> Dict[str, Any]:
        """
        Train the model on the task.
        
        Args:
            training_args: Optional TrainingArguments (uses defaults if None)
            **training_kwargs: Arguments to pass to get_training_arguments()
            
        Returns:
            Dictionary with training results and metrics
        """
        # Load model and tokenizer if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        # Load dataset if not already loaded
        if self.datasets is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # Get training arguments
        if training_args is None:
            training_args = self.get_training_arguments(**training_kwargs)
        
        # Preprocess datasets
        print("\nPreprocessing datasets...")
        tokenized_datasets = self.datasets.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.datasets["train"].column_names,
            desc="Tokenizing"
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets.get("validation") or tokenized_datasets.get("test"),
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        train_result = trainer.train()
        
        # Evaluate
        print("\n" + "="*80)
        print("Evaluating")
        print("="*80)
        eval_result = trainer.evaluate()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Compile results
        results = {
            "model": self.model_name,
            "task": self.__class__.__name__,
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "eval_metrics": eval_result,
            "training_args": training_args.to_dict(),
        }
        
        # Save results to JSON
        results_path = os.path.join(self.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to {results_path}")
        
        return results
    
    def evaluate_only(
        self,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model without training.
        
        Args:
            checkpoint_path: Path to model checkpoint (uses self.output_dir if None)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Load model
        if checkpoint_path:
            print(f"Loading checkpoint: {checkpoint_path}")
            self.model = self.prepare_model(
                AutoModel.from_pretrained(checkpoint_path),
                AutoConfig.from_pretrained(checkpoint_path)
            )
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        else:
            self.load_model_and_tokenizer()
        
        self.model.to(self.device)
        
        # Load dataset if not already loaded
        if self.datasets is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # Preprocess datasets
        print("\nPreprocessing datasets...")
        tokenized_datasets = self.datasets.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.datasets["train"].column_names,
            desc="Tokenizing"
        )
        
        # Create trainer for evaluation
        eval_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_eval_batch_size=16,
            fp16=self.device == "cuda",
            report_to="none",
        )
        
        trainer = Trainer(
            model=self.model,
            args=eval_args,
            eval_dataset=tokenized_datasets.get("validation") or tokenized_datasets.get("test"),
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        
        # Evaluate
        print("\n" + "="*80)
        print("Evaluating")
        print("="*80)
        eval_result = trainer.evaluate()
        
        # Compile results
        results = {
            "model": self.model_name,
            "task": self.__class__.__name__,
            "eval_metrics": eval_result,
        }
        
        # Save results to JSON
        results_path = os.path.join(self.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to {results_path}")
        
        return results

