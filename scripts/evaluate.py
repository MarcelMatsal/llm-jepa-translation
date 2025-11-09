"""
Evaluation script for dual-objective BERT model.
Computes CLS alignment metrics, discrimination tests, and retrieval accuracy.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizer
import argparse
import json

from src.models.bert_dual_objective import BertDualObjective
from src.data.datasets import load_translation_dataset
from src.data.collators import SimpleCollator
from src.training.metrics import (
    evaluate_model_comprehensive,
    compute_discrimination_score,
    compute_retrieval_accuracy
)


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
    
    # Load model
    print("\n" + "="*80)
    print("Loading Model")
    print("="*80)
    
    if args.checkpoint:
        print(f"Loading from checkpoint: {args.checkpoint}")
        # Check if it's a HuggingFace-style directory
        if os.path.isdir(args.checkpoint):
            model = BertDualObjective.from_pretrained(args.checkpoint)
        else:
            # Load from .pt checkpoint
            model = BertDualObjective(
                model_name=config['model']['base_model'],
                lambda_alignment=config['model']['lambda_alignment'],
                alignment_loss_type=config['model']['alignment_loss_type']
            )
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print("Loading pre-trained model (no fine-tuning)")
        model = BertDualObjective(
            model_name=config['model']['base_model'],
            lambda_alignment=config['model']['lambda_alignment'],
            alignment_loss_type=config['model']['alignment_loss_type']
        )
    
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(config['model']['base_model'])
    
    # Determine which language pairs to evaluate
    if args.lang_pair:
        eval_lang_pairs = [args.lang_pair]
    else:
        eval_lang_pairs = config.get('evaluation', {}).get('eval_lang_pairs', config['data']['lang_pairs'])
    
    print(f"\nEvaluating on language pairs: {eval_lang_pairs}")
    
    # Load datasets and create dataloaders
    print("\n" + "="*80)
    print("Loading Evaluation Datasets")
    print("="*80)
    
    eval_dataloaders = {}
    collator = SimpleCollator(tokenizer=tokenizer, max_length=config['data']['max_length'])
    
    for lang_pair in eval_lang_pairs:
        print(f"\nLoading {lang_pair}...")
        try:
            dataset = load_translation_dataset(
                lang_pair=lang_pair,
                split='validation' if not args.use_train else 'train',
                max_examples=args.max_examples,
                min_length=config['data'].get('min_text_length', 10),
                max_length=config['data'].get('max_text_length', 500)
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=config.get('evaluation', {}).get('eval_batch_size', 32),
                shuffle=False,
                collate_fn=collator,
                num_workers=config['data'].get('num_workers', 0)
            )
            
            eval_dataloaders[lang_pair] = dataloader
            print(f"  Loaded {len(dataset)} examples")
        except Exception as e:
            print(f"  Warning: Failed to load {lang_pair}: {e}")
    
    if len(eval_dataloaders) == 0:
        print("Error: No datasets loaded successfully")
        return
    
    # Run comprehensive evaluation
    print("\n" + "="*80)
    print("Running Evaluation")
    print("="*80)
    
    results = evaluate_model_comprehensive(
        model=model,
        dataloaders=eval_dataloaders,
        device=device
    )
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    for lang_pair, metrics in results.items():
        print(f"\n{lang_pair.upper()}:")
        print(f"  CLS Cosine Similarity:     {metrics['cosine_sim_mean']:.4f} ± {metrics['cosine_sim_std']:.4f}")
        print(f"  CLS Euclidean Distance:    {metrics['euclidean_dist_mean']:.4f} ± {metrics['euclidean_dist_std']:.4f}")
        print(f"  Discrimination Score:      {metrics['discrimination_score']:.4f}")
        print(f"    - Translation pairs:     {metrics['translation_sim_mean']:.4f}")
        print(f"    - Random pairs:          {metrics['random_sim_mean']:.4f}")
        print(f"  Retrieval Accuracy@1:      {metrics['retrieval_accuracy_at_1_average']:.4f}")
        print(f"  Number of examples:        {metrics['num_examples']}")
    
    # Compute averages across language pairs
    avg_cosine_sim = sum(r['cosine_sim_mean'] for r in results.values()) / len(results)
    avg_discrimination = sum(r['discrimination_score'] for r in results.values()) / len(results)
    avg_retrieval = sum(r['retrieval_accuracy_at_1_average'] for r in results.values()) / len(results)
    
    print(f"\nAVERAGE ACROSS ALL PAIRS:")
    print(f"  CLS Cosine Similarity:     {avg_cosine_sim:.4f}")
    print(f"  Discrimination Score:      {avg_discrimination:.4f}")
    print(f"  Retrieval Accuracy@1:      {avg_retrieval:.4f}")
    
    # Save results
    if args.output:
        print(f"\nSaving results to {args.output}")
        
        # Convert results to serializable format
        serializable_results = {}
        for lang_pair, metrics in results.items():
            serializable_results[lang_pair] = {k: float(v) if isinstance(v, (int, float)) else v 
                                               for k, v in metrics.items()}
        
        # Add summary
        serializable_results['summary'] = {
            'avg_cosine_similarity': float(avg_cosine_sim),
            'avg_discrimination_score': float(avg_discrimination),
            'avg_retrieval_accuracy': float(avg_retrieval),
            'checkpoint': args.checkpoint if args.checkpoint else 'pretrained',
            'lang_pairs': eval_lang_pairs
        }
        
        with open(args.output, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved!")
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate dual-objective BERT model')
    parser.add_argument(
        '--config',
        type=str,
        default='experiments/exp_test/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (directory or .pt file)'
    )
    parser.add_argument(
        '--lang_pair',
        type=str,
        default=None,
        help='Single language pair to evaluate (e.g., "de-en"). If not specified, uses config.'
    )
    parser.add_argument(
        '--max_examples',
        type=int,
        default=1000,
        help='Maximum number of examples per language pair'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save evaluation results (JSON format)'
    )
    parser.add_argument(
        '--use_train',
        action='store_true',
        help='Use training split instead of validation'
    )
    
    args = parser.parse_args()
    main(args)
