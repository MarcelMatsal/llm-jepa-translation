"""
Main training script for Multilingual JEPA.
Uses YAML config files for all configuration.
"""
import argparse
import torch
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from omegaconf import OmegaConf
from src.models import MultilingualJEPA
from src.data import get_dataset, get_dataloader
from src.training import Trainer, compute_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Multilingual JEPA')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (YAML, default: config.yaml)')
    args = parser.parse_args()
    
    # Load config file
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    print(f'Loading config from {args.config}')
    cfg = OmegaConf.load(args.config)
    
    # Validate required config
    if cfg.data.lang_pair is None:
        raise ValueError("Language pair must be specified in config file")
    
    # Print config
    print('\n' + '=' * 60)
    print('Configuration:')
    print('=' * 60)
    print(OmegaConf.to_yaml(cfg))
    print('=' * 60 + '\n')
    
    # Device
    device = cfg.output.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('Warning: CUDA not available, using CPU')
        device = 'cpu'
    
    # Language mapping
    lang_map = {'en': 0, 'fr': 1, 'de': 2, 'es': 3, 'it': 4, 'pt': 5, 'ru': 6, 'zh': 7, 'ja': 8}
    num_languages = max(cfg.model.num_languages, len(set(lang_map.values())))
    
    # Load datasets
    print(f'Loading WMT19 dataset: {cfg.data.lang_pair}')
    train_dataset = get_dataset(cfg.data.lang_pair, lang_map, split='train')
    
    # Load validation split
    val_dataset = None
    try:
        val_dataset = get_dataset(cfg.data.lang_pair, lang_map, split='validation')
    except:
        print('Warning: Validation split not available')
    
    if len(train_dataset) == 0:
        raise ValueError(f"No training samples found for lang_pair {cfg.data.lang_pair}")
    
    print(f'Train samples: {len(train_dataset)}')
    if val_dataset and len(val_dataset) > 0:
        print(f'Val samples: {len(val_dataset)}')
    else:
        print('Warning: No validation dataset found')
    
    train_loader = get_dataloader(
        train_dataset, 
        batch_size=cfg.data.batch_size, 
        shuffle=True, 
        num_workers=cfg.data.num_workers
    )
    
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = get_dataloader(
            val_dataset, 
            batch_size=cfg.data.batch_size, 
            shuffle=False, 
            num_workers=cfg.data.num_workers
        )
    
    # Model
    print('Initializing model...')
    model = MultilingualJEPA(
        encoder_name=cfg.model.encoder_name,
        pooling=cfg.model.pooling,
        num_languages=num_languages,
        tau=cfg.model.tau
    )
    
    # Optimizer
    trainable_params = list(model.x_encoder.parameters()) + \
                       list(model.predictor.parameters()) + \
                       list(model.lang_embedding.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.training.learning_rate)
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        max_grad_norm=cfg.training.max_grad_norm,
        log_interval=cfg.training.log_interval
    )
    
    # Train
    print('Starting training...')
    trainer.train(num_epochs=cfg.training.epochs)
    
    # Evaluate if validation set available
    if val_loader:
        print('\nComputing final metrics...')
        metrics = compute_metrics(model, val_loader, device=device)
        print(f'Cosine Similarity: {metrics["cosine_similarity"]:.4f}')
        print(f'MSE: {metrics["mse"]:.4f}')
        print(f'Embedding Diversity: {metrics["embedding_diversity"]:.4f}')
        print(f'Linearity Error: {metrics["linearity_error"]:.4f}')
    else:
        metrics = {}
    
    # Save model
    os.makedirs(cfg.output.save_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': OmegaConf.to_container(cfg, resolve=True)
    }, os.path.join(cfg.output.save_dir, 'checkpoint.pt'))
    print(f'\nModel saved to {cfg.output.save_dir}/checkpoint.pt')
    
    # Save config used for this run
    OmegaConf.save(cfg, os.path.join(cfg.output.save_dir, 'config.yaml'))
    print(f'Config saved to {cfg.output.save_dir}/config.yaml')


if __name__ == '__main__':
    main()

