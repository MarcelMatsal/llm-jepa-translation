"""
Default configurations and dataset mappings for downstream tasks.
"""
from typing import Dict, Any, Optional


# Default dataset mappings per task type
TASK_DATASETS = {
    "sequence_classification": {
        "glue": {
            "sst2": {"name": "glue", "config": "sst2", "text_column": "sentence", "label_column": "label"},
            "mrpc": {"name": "glue", "config": "mrpc", "text_column": "sentence1", "text2_column": "sentence2", "label_column": "label"},
            "cola": {"name": "glue", "config": "cola", "text_column": "sentence", "label_column": "label"},
            "qqp": {"name": "glue", "config": "qqp", "text_column": "question1", "text2_column": "question2", "label_column": "label"},
            "mnli": {"name": "glue", "config": "mnli", "text_column": "premise", "text2_column": "hypothesis", "label_column": "label"},
            "qnli": {"name": "glue", "config": "qnli", "text_column": "question", "text2_column": "sentence", "label_column": "label"},
            "rte": {"name": "glue", "config": "rte", "text_column": "sentence1", "text2_column": "sentence2", "label_column": "label"},
            "wnli": {"name": "glue", "config": "wnli", "text_column": "sentence1", "text2_column": "sentence2", "label_column": "label"},
        },
        "imdb": {
            "default": {"name": "imdb", "text_column": "text", "label_column": "label"},
        },
    },
    "token_classification": {
        "conll2003": {
            "default": {
                "name": "conll2003",
                "text_column": "tokens",
                "label_column": "ner_tags",
            },
        },
    },
    "question_answering": {
        "squad": {
            "v1_1": {"name": "squad", "text_column": "context", "question_column": "question", "answer_column": "answers"},
        },
    },
    "multiple_choice": {
        "swag": {
            "default": {"name": "swag", "context_column": "sent1", "ending_column": "ending", "label_column": "label"},
        },
        "race": {
            "all": {"name": "race", "config": "all", "context_column": "article", "question_column": "question", "options_column": "options", "label_column": "answer"},
        },
    },
}


# Default hyperparameters per task type
TASK_HYPERPARAMETERS = {
    "sequence_classification": {
        "num_epochs": 3,
        "learning_rate": 2e-5,
        "batch_size": 16,
        "eval_batch_size": 16,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "max_length": 128,
        "metric_for_best_model": "eval_accuracy",
        "greater_is_better": True,
    },
    "token_classification": {
        "num_epochs": 3,
        "learning_rate": 2e-5,
        "batch_size": 16,
        "eval_batch_size": 16,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "max_length": 128,
        "metric_for_best_model": "eval_f1",
        "greater_is_better": True,
    },
    "question_answering": {
        "num_epochs": 3,
        "learning_rate": 3e-5,
        "batch_size": 16,
        "eval_batch_size": 16,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "max_length": 384,
        "doc_stride": 128,
        "metric_for_best_model": "eval_f1",
        "greater_is_better": True,
    },
    "multiple_choice": {
        "num_epochs": 3,
        "learning_rate": 2e-5,
        "batch_size": 8,  # Smaller batch size due to multiple choices per example
        "eval_batch_size": 8,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "max_length": 128,
        "metric_for_best_model": "eval_accuracy",
        "greater_is_better": True,
    },
}


def get_dataset_config(
    task_type: str,
    dataset_name: str,
    dataset_config: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get dataset configuration for a task.
    
    Args:
        task_type: Type of task (sequence_classification, token_classification, etc.)
        dataset_name: Name of dataset (glue, conll2003, squad, etc.)
        dataset_config: Optional dataset configuration (sst2, v1_1, etc.)
        
    Returns:
        Dictionary with dataset configuration
    """
    if task_type not in TASK_DATASETS:
        raise ValueError(f"Unknown task type: {task_type}")
    
    if dataset_name not in TASK_DATASETS[task_type]:
        raise ValueError(f"Unknown dataset '{dataset_name}' for task '{task_type}'")
    
    datasets = TASK_DATASETS[task_type][dataset_name]
    
    # If no config specified, use default
    if dataset_config is None:
        if "default" in datasets:
            return datasets["default"]
        else:
            # Return first available config
            return list(datasets.values())[0]
    
    if dataset_config not in datasets:
        raise ValueError(
            f"Unknown dataset config '{dataset_config}' for dataset '{dataset_name}'. "
            f"Available: {list(datasets.keys())}"
        )
    
    return datasets[dataset_config]


def get_hyperparameters(task_type: str) -> Dict[str, Any]:
    """
    Get default hyperparameters for a task type.
    
    Args:
        task_type: Type of task
        
    Returns:
        Dictionary with hyperparameters
    """
    if task_type not in TASK_HYPERPARAMETERS:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return TASK_HYPERPARAMETERS[task_type].copy()


def list_available_datasets(task_type: str) -> Dict[str, list]:
    """
    List all available datasets for a task type.
    
    Args:
        task_type: Type of task
        
    Returns:
        Dictionary mapping dataset names to available configs
    """
    if task_type not in TASK_DATASETS:
        raise ValueError(f"Unknown task type: {task_type}")
    
    result = {}
    for dataset_name, configs in TASK_DATASETS[task_type].items():
        result[dataset_name] = list(configs.keys())
    
    return result

