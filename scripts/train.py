"""
Main training script for Multilingual JEPA.
"""
import argparse
import torch
from src.models import MultilingualJEPA
from src.data import get_dataset, get_dataloader
from src.training import Trainer, compute_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Multilingual JEPA')
    
    # Model args
    parser.add_argument('--encoder_name', type=str, default='bert-base-multilingual-cased',
                       help='HuggingFace model name')
    parser.add_argument('--pooling', type=str, default='cls', choices=['cls', 'mean', 'attention'],
                       help='Pooling strategy')
    parser.add_argument('--tau', type=float, default=0.999, help='EMA decay rate')
    
    # Data args
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (HF dataset or path to JSONL)')
    parser.add_argument('--lang_pair', type=str, default='en-de',
                       help='Language pair (e.g., en-de)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # Training args
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Output args
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_interval', type=int, default=100)
    
    args = parser.parse_args()
    
    # Device
    device = args.device
    
    # Language mapping
    lang_map = {'en': 0, 'fr': 1, 'de': 2, 'es': 3, 'it': 4, 'pt': 5, 'ru': 6, 'zh': 7, 'ja': 8}
    num_languages = len(set(lang_map.values()))
    
    # Load datasets
    print(f'Loading dataset: {args.dataset}')
    train_dataset = get_dataset(args.dataset, args.lang_pair, lang_map, split='train')
    val_dataset = get_dataset(args.dataset, args.lang_pair, lang_map, split='validation')
    
    train_loader = get_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = get_dataloader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    
    # Model
    print('Initializing model...')
    model = MultilingualJEPA(
        encoder_name=args.encoder_name,
        pooling=args.pooling,
        num_languages=num_languages,
        tau=args.tau
    )
    
    # Optimizer
    trainable_params = list(model.x_encoder.parameters()) + \
                       list(model.predictor.parameters()) + \
                       list(model.lang_embedding.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval
    )
    
    # Train
    print('Starting training...')
    trainer.train(num_epochs=args.epochs)
    
    # Evaluate
    print('\nComputing final metrics...')
    metrics = compute_metrics(model, val_loader, device=device)
    print(f'Cosine Similarity: {metrics["cosine_similarity"]:.4f}')
    print(f'MSE: {metrics["mse"]:.4f}')
    print(f'Embedding Diversity: {metrics["embedding_diversity"]:.4f}')
    print(f'Linearity Error: {metrics["linearity_error"]:.4f}')
    
    # Save model
    import os
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, f'{args.save_dir}/checkpoint.pt')
    print(f'\nModel saved to {args.save_dir}/checkpoint.pt')


if __name__ == '__main__':
    main()

