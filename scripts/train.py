"""
Training script for dual-objective BERT model.
Loads configuration, creates model and data loaders, and trains the model.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizer, get_linear_schedule_with_warmup
import argparse
import wandb

from src.models.bert_dual_objective import BertDualObjective
from src.data.datasets import load_multilingual_dataset
from src.data.collators import DualObjectiveCollator, SimpleCollator
from src.training.trainer import DualObjectiveTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Initialize Weights & Biases
    use_wandb = config.get('wandb', {}).get('enabled', True) and not args.no_wandb
    if use_wandb:
        wandb.init(
            entity=config.get('wandb', {}).get('entity', None),
            project=config.get('wandb', {}).get('project', 'llm-jepa-translation'),
            name=config.get('wandb', {}).get('run_name', None),
            config=config,
            tags=config.get('wandb', {}).get('tags', []),
            notes=config.get('wandb', {}).get('notes', ''),
            resume='allow' if args.resume else None
        )
        print(f"✓ W&B initialized: {wandb.run.name}")
        print(f"  URL: {wandb.run.url}")
    else:
        print("W&B logging disabled")
    
    # Set device
    device = config['output']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {config['model']['base_model']}")
    tokenizer = XLMRobertaTokenizer.from_pretrained(config['model']['base_model'])
    
    # Load datasets
    print("\n" + "="*80)
    print("Loading Datasets")
    print("="*80)
    
    # Get dataset source (default to 'opus100' for backward compatibility)
    dataset_source = config['data'].get('dataset_source', 'opus100')
    train_split = config['data'].get('train_split', 'train')
    
    train_dataset = load_multilingual_dataset(
        lang_pairs=config['data']['lang_pairs'],
        split=train_split,
        max_examples_per_pair=config['data'].get('max_examples_per_pair'),
        min_length=config['data'].get('min_text_length', 10),
        max_length=config['data'].get('max_text_length', 500),
        dataset_source=dataset_source
    )
    
    print(f"\nTotal training examples: {len(train_dataset)}")
    
    # Create validation dataset (if available)
    val_dataset = None
    val_split = config['data'].get('val_split', 'validation')
    try:
        val_dataset = load_multilingual_dataset(
            lang_pairs=config['data']['lang_pairs'],
            split=val_split,
            max_examples_per_pair=1000,  # Limit validation size
            min_length=config['data'].get('min_text_length', 10),
            max_length=config['data'].get('max_text_length', 500),
            dataset_source=dataset_source
        )
        print(f"Total validation examples: {len(val_dataset)}")
    except Exception as e:
        print(f"Warning: Could not load validation dataset: {e}")
    
    # Create collators
    train_collator = DualObjectiveCollator(
        tokenizer=tokenizer,
        mlm_probability=config['model']['mlm_probability'],
        max_length=config['data']['max_length'],
        mlm_strategy=config['model'].get('mlm_strategy', 'monolingual')
    )
    
    # Use same collator for validation (needed for compute_total_loss)
    val_collator = DualObjectiveCollator(
        tokenizer=tokenizer,
        mlm_probability=config['model']['mlm_probability'],
        max_length=config['data']['max_length'],
        mlm_strategy=config['model'].get('mlm_strategy', 'monolingual')
    ) if val_dataset is not None else None
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        collate_fn=train_collator,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=(device == 'cuda')
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            collate_fn=val_collator,
            num_workers=config['data'].get('num_workers', 0),
            pin_memory=(device == 'cuda')
        )
    
    # Create model
    print("\n" + "="*80)
    print("Initializing Model")
    print("="*80)
    
    # Check if using new or old config format
    if 'alignment_loss' in config['model']:
        # New format: alignment_loss is a dict
        model = BertDualObjective(
            model_name=config['model']['base_model'],
            lambda_alignment=config['model']['lambda_alignment'],
            alignment_loss_config=config['model']['alignment_loss']
        )
        print(f"Using alignment loss: {config['model']['alignment_loss']['type']}")
    else:
        # Old format: alignment_loss_type is a string
        model = BertDualObjective(
            model_name=config['model']['base_model'],
            lambda_alignment=config['model']['lambda_alignment'],
            alignment_loss_type=config['model']['alignment_loss_type']
        )
        print(f"Using alignment loss (deprecated format): {config['model']['alignment_loss_type']}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    num_training_steps = len(train_loader) * config['training']['epochs']
    num_warmup_steps = config['training'].get('warmup_steps', 0)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Create trainer
    print("\n" + "="*80)
    print("Creating Trainer")
    print("="*80)
    
    # Get HuggingFace Hub configuration
    hub_model_id = args.hub_model_id or config.get('output', {}).get('hub_model_id')
    hub_model_base = config.get('output', {}).get('hub_model_base', 'maktzgls/bert-jepa')
    experiment_name = args.experiment_name or config.get('experiment_name')
    push_to_hub = args.push_to_hub or config.get('output', {}).get('push_to_hub', False)
    
    # Auto-generate hub_model_id if experiment_name is provided but hub_model_id is not
    if push_to_hub and not hub_model_id and experiment_name:
        hub_model_id = f"{hub_model_base}-{experiment_name}"
        print(f"✓ Auto-generated Hub model ID: {hub_model_id}")
    
    if push_to_hub and hub_model_id:
        print(f"✓ HuggingFace Hub push enabled")
        print(f"  Model ID: {hub_model_id}")
        if experiment_name:
            print(f"  Experiment: {experiment_name}")
    
    # Prepare experiment metadata
    # Handle both old and new config formats for alignment loss
    if 'alignment_loss' in config['model']:
        alignment_loss_info = config['model']['alignment_loss']['type']
    else:
        alignment_loss_info = config['model'].get('alignment_loss_type', 'unknown')
    
    experiment_metadata = {
        'experiment_name': experiment_name,
        'config_path': args.config,
        'config_description': config.get('description', ''),
        'base_model': config['model']['base_model'],
        'lambda_alignment': config['model']['lambda_alignment'],
        'alignment_loss_type': alignment_loss_info,
        'lang_pairs': config['data']['lang_pairs'],
        'batch_size': config['data']['batch_size'],
        'epochs': config['training']['epochs']
    }
    
    trainer = DualObjectiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_grad_norm=config['training']['max_grad_norm'],
        log_interval=config['training']['log_interval'],
        save_dir=config['output']['save_dir'],
        accumulation_steps=config['training'].get('accumulation_steps', 1),
        use_wandb=use_wandb,
        use_amp=True,  # Enable mixed precision for memory efficiency
        use_gradient_checkpointing=True,  # Enable gradient checkpointing
        tokenizer=tokenizer,
        hub_model_id=hub_model_id,
        push_to_hub=push_to_hub,
        experiment_metadata=experiment_metadata
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    trainer.train(
        num_epochs=config['training']['epochs'],
        save_every=config['training'].get('save_every', 1)
    )
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Checkpoints saved to: {config['output']['save_dir']}")
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()
        print("W&B run finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train dual-objective BERT model')
    parser.add_argument(
        '--config',
        type=str,
        default='experiments/exp_test/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    parser.add_argument(
        '--hub-model-id',
        type=str,
        default=None,
        help='HuggingFace Hub model ID (e.g., username/model-name). If not provided, will be auto-generated from experiment_name'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name (e.g., "small", "medium"). Used to auto-generate hub_model_id if not explicitly provided'
    )
    parser.add_argument(
        '--push-to-hub',
        action='store_true',
        help='Push model checkpoints to HuggingFace Hub'
    )
    
    args = parser.parse_args()
    main(args)
