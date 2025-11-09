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
    
    train_dataset = load_multilingual_dataset(
        lang_pairs=config['data']['lang_pairs'],
        split='train',
        max_examples_per_pair=config['data'].get('max_examples_per_pair'),
        min_length=config['data'].get('min_text_length', 10),
        max_length=config['data'].get('max_text_length', 500)
    )
    
    print(f"\nTotal training examples: {len(train_dataset)}")
    
    # Create validation dataset (if available)
    val_dataset = None
    try:
        val_dataset = load_multilingual_dataset(
            lang_pairs=config['data']['lang_pairs'],
            split='validation',
            max_examples_per_pair=1000,  # Limit validation size
            min_length=config['data'].get('min_text_length', 10),
            max_length=config['data'].get('max_text_length', 500)
        )
        print(f"Total validation examples: {len(val_dataset)}")
    except Exception as e:
        print(f"Warning: Could not load validation dataset: {e}")
    
    # Create collators
    train_collator = DualObjectiveCollator(
        tokenizer=tokenizer,
        mlm_probability=config['model']['mlm_probability'],
        max_length=config['data']['max_length']
    )
    
    val_collator = SimpleCollator(
        tokenizer=tokenizer,
        max_length=config['data']['max_length']
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
    
    model = BertDualObjective(
        model_name=config['model']['base_model'],
        lambda_alignment=config['model']['lambda_alignment'],
        alignment_loss_type=config['model']['alignment_loss_type']
    )
    
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
        accumulation_steps=config['training'].get('accumulation_steps', 1)
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
    
    args = parser.parse_args()
    main(args)
