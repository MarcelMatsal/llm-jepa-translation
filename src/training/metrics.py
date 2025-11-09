"""
Evaluation metrics for dual-objective BERT training.
Includes CLS similarity, discrimination tests, and alignment quality metrics.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import random


def compute_cls_similarity(
    cls1: torch.Tensor,
    cls2: torch.Tensor,
    metric: str = 'cosine'
) -> float:
    """
    Compute similarity between CLS embeddings.
    
    Args:
        cls1: First CLS embeddings (batch_size, d_model)
        cls2: Second CLS embeddings (batch_size, d_model)
        metric: 'cosine' or 'euclidean'
    
    Returns:
        Similarity score (higher = more similar for cosine, lower = more similar for euclidean)
    """
    if metric == 'cosine':
        # Normalize and compute cosine similarity
        cls1_norm = F.normalize(cls1, p=2, dim=-1)
        cls2_norm = F.normalize(cls2, p=2, dim=-1)
        similarity = F.cosine_similarity(cls1_norm, cls2_norm, dim=-1).mean().item()
        return similarity
    
    elif metric == 'euclidean':
        # Compute Euclidean distance
        cls1_norm = F.normalize(cls1, p=2, dim=-1)
        cls2_norm = F.normalize(cls2, p=2, dim=-1)
        distance = torch.norm(cls1_norm - cls2_norm, p=2, dim=-1).mean().item()
        return distance
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_alignment_metrics(
    cls1: torch.Tensor,
    cls2: torch.Tensor
) -> Dict[str, float]:
    """
    Compute multiple alignment metrics between CLS embeddings.
    
    Args:
        cls1: First CLS embeddings (batch_size, d_model)
        cls2: Second CLS embeddings (batch_size, d_model)
    
    Returns:
        Dictionary with various alignment metrics
    """
    # Normalize embeddings
    cls1_norm = F.normalize(cls1, p=2, dim=-1)
    cls2_norm = F.normalize(cls2, p=2, dim=-1)
    
    # Cosine similarity
    cosine_sim = F.cosine_similarity(cls1_norm, cls2_norm, dim=-1)
    
    # Euclidean distance
    euclidean_dist = torch.norm(cls1_norm - cls2_norm, p=2, dim=-1)
    
    # MSE
    mse = F.mse_loss(cls1_norm, cls2_norm, reduction='none').mean(dim=-1)
    
    metrics = {
        'cosine_sim_mean': cosine_sim.mean().item(),
        'cosine_sim_std': cosine_sim.std().item(),
        'euclidean_dist_mean': euclidean_dist.mean().item(),
        'euclidean_dist_std': euclidean_dist.std().item(),
        'mse_mean': mse.mean().item(),
        'mse_std': mse.std().item()
    }
    
    return metrics


def compute_discrimination_score(
    model,
    dataloader,
    device: str = 'cuda',
    num_batches: int = 10
) -> Dict[str, float]:
    """
    Compute discrimination ability: how well the model distinguishes
    translation pairs from random pairs.
    
    Args:
        model: BertDualObjective model
        dataloader: DataLoader with translation pairs
        device: Device to run on
        num_batches: Number of batches to evaluate
    
    Returns:
        Dictionary with discrimination metrics
    """
    model.eval()
    
    translation_similarities = []
    random_similarities = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            positions_dict = batch['positions_dict']
            
            # Extract CLS embeddings for translation pairs
            cls1, cls2 = model.get_cls_embeddings_for_eval(
                input_ids, attention_mask, positions_dict
            )
            
            # Compute similarity for translation pairs
            trans_sim = F.cosine_similarity(cls1, cls2, dim=-1)
            translation_similarities.extend(trans_sim.cpu().numpy())
            
            # Create random pairs by shuffling cls2
            batch_size = cls2.shape[0]
            if batch_size > 1:
                # Shuffle to create random pairs
                indices = torch.randperm(batch_size, device=device)
                cls2_shuffled = cls2[indices]
                
                # Compute similarity for random pairs
                random_sim = F.cosine_similarity(cls1, cls2_shuffled, dim=-1)
                random_similarities.extend(random_sim.cpu().numpy())
    
    translation_similarities = np.array(translation_similarities)
    random_similarities = np.array(random_similarities)
    
    # Discrimination score: difference between translation and random similarities
    discrimination = translation_similarities.mean() - random_similarities.mean()
    
    metrics = {
        'translation_sim_mean': translation_similarities.mean(),
        'translation_sim_std': translation_similarities.std(),
        'random_sim_mean': random_similarities.mean(),
        'random_sim_std': random_similarities.std(),
        'discrimination_score': discrimination,
        'num_examples': len(translation_similarities)
    }
    
    return metrics


def evaluate_language_pair_alignment(
    model,
    dataloader,
    device: str = 'cuda',
    lang_pair: str = 'unknown'
) -> Dict[str, float]:
    """
    Evaluate alignment quality for a specific language pair.
    
    Args:
        model: BertDualObjective model
        dataloader: DataLoader for the language pair
        device: Device to run on
        lang_pair: Language pair identifier (for logging)
    
    Returns:
        Dictionary with alignment metrics
    """
    model.eval()
    
    all_cls1 = []
    all_cls2 = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {lang_pair}"):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            positions_dict = batch['positions_dict']
            
            # Extract CLS embeddings
            cls1, cls2 = model.get_cls_embeddings_for_eval(
                input_ids, attention_mask, positions_dict
            )
            
            all_cls1.append(cls1.cpu())
            all_cls2.append(cls2.cpu())
    
    # Concatenate all embeddings
    all_cls1 = torch.cat(all_cls1, dim=0)
    all_cls2 = torch.cat(all_cls2, dim=0)
    
    # Compute alignment metrics
    metrics = compute_alignment_metrics(all_cls1, all_cls2)
    metrics['lang_pair'] = lang_pair
    metrics['num_examples'] = all_cls1.shape[0]
    
    return metrics


def compute_retrieval_accuracy(
    model,
    dataloader,
    device: str = 'cuda',
    top_k: int = 1
) -> Dict[str, float]:
    """
    Compute cross-lingual retrieval accuracy.
    For each lang1 sentence, find the correct lang2 translation among all options.
    
    Args:
        model: BertDualObjective model
        dataloader: DataLoader with translation pairs
        device: Device to run on
        top_k: Compute accuracy at top-k (default: 1)
    
    Returns:
        Dictionary with retrieval metrics
    """
    model.eval()
    
    all_cls1 = []
    all_cls2 = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            positions_dict = batch['positions_dict']
            
            # Extract CLS embeddings
            cls1, cls2 = model.get_cls_embeddings_for_eval(
                input_ids, attention_mask, positions_dict
            )
            
            all_cls1.append(cls1.cpu())
            all_cls2.append(cls2.cpu())
    
    # Concatenate all embeddings
    all_cls1 = torch.cat(all_cls1, dim=0)  # (N, d_model)
    all_cls2 = torch.cat(all_cls2, dim=0)  # (N, d_model)
    
    N = all_cls1.shape[0]
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(all_cls1, all_cls2.T)  # (N, N)
    
    # For each query (row), the correct match is on the diagonal
    correct_indices = torch.arange(N)
    
    # Get top-k predictions for each query
    top_k_indices = torch.topk(similarity_matrix, k=min(top_k, N), dim=1).indices
    
    # Check if correct index is in top-k
    correct_in_top_k = (top_k_indices == correct_indices.unsqueeze(1)).any(dim=1)
    
    accuracy_at_k = correct_in_top_k.float().mean().item()
    
    # Also compute accuracy in both directions
    similarity_matrix_t = similarity_matrix.T
    top_k_indices_reverse = torch.topk(similarity_matrix_t, k=min(top_k, N), dim=1).indices
    correct_in_top_k_reverse = (top_k_indices_reverse == correct_indices.unsqueeze(1)).any(dim=1)
    accuracy_at_k_reverse = correct_in_top_k_reverse.float().mean().item()
    
    metrics = {
        f'retrieval_accuracy_at_{top_k}_lang1_to_lang2': accuracy_at_k,
        f'retrieval_accuracy_at_{top_k}_lang2_to_lang1': accuracy_at_k_reverse,
        f'retrieval_accuracy_at_{top_k}_average': (accuracy_at_k + accuracy_at_k_reverse) / 2,
        'num_examples': N
    }
    
    return metrics


def evaluate_model_comprehensive(
    model,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    device: str = 'cuda'
) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive evaluation across multiple language pairs and metrics.
    
    Args:
        model: BertDualObjective model
        dataloaders: Dictionary mapping lang_pair -> DataLoader
        device: Device to run on
    
    Returns:
        Dictionary mapping lang_pair -> metrics
    """
    all_results = {}
    
    for lang_pair, dataloader in dataloaders.items():
        print(f"\nEvaluating {lang_pair}...")
        
        # Alignment metrics
        alignment_metrics = evaluate_language_pair_alignment(
            model, dataloader, device, lang_pair
        )
        
        # Discrimination metrics
        discrimination_metrics = compute_discrimination_score(
            model, dataloader, device, num_batches=10
        )
        
        # Retrieval metrics
        retrieval_metrics = compute_retrieval_accuracy(
            model, dataloader, device, top_k=1
        )
        
        # Combine all metrics
        combined_metrics = {
            **alignment_metrics,
            **discrimination_metrics,
            **retrieval_metrics
        }
        
        all_results[lang_pair] = combined_metrics
        
        # Print summary
        print(f"  Cosine similarity: {combined_metrics['cosine_sim_mean']:.4f}")
        print(f"  Discrimination score: {combined_metrics['discrimination_score']:.4f}")
        print(f"  Retrieval accuracy: {combined_metrics['retrieval_accuracy_at_1_average']:.4f}")
    
    return all_results
