"""
Evaluation script for Multilingual JEPA.
"""
import argparse
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import MultilingualJEPA
from src.data import get_dataset, get_dataloader
from src.training import compute_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Multilingual JEPA')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--lang_pair', type=str, required=True, help='Language pair (e.g., en-de)')
    parser.add_argument('--encoder_name', type=str, default='bert-base-multilingual-cased')
    parser.add_argument('--pooling', type=str, default='cls')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Language mapping
    lang_map = {'en': 0, 'fr': 1, 'de': 2, 'es': 3, 'it': 4, 'pt': 5, 'ru': 6, 'zh': 7, 'ja': 8}
    num_languages = len(set(lang_map.values()))
    
    # Load dataset (WMT19)
    print(f'Loading WMT19 dataset: {args.lang_pair}')
    test_dataset = get_dataset(args.lang_pair, lang_map, split='validation')
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load model
    model = MultilingualJEPA(
        encoder_name=args.encoder_name,
        pooling=args.pooling,
        num_languages=num_languages
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    
    # Evaluate
    metrics = compute_metrics(model, test_loader, device=args.device)
    
    print('\nEvaluation Metrics:')
    print(f'Cosine Similarity: {metrics["cosine_similarity"]:.4f}')
    print(f'MSE: {metrics["mse"]:.4f}')
    print(f'Embedding Diversity: {metrics["embedding_diversity"]:.4f}')
    print(f'Linearity Error: {metrics["linearity_error"]:.4f}')
    print(f'Top 10 Singular Values: {metrics["top_singular_values"]}')


if __name__ == '__main__':
    main()

