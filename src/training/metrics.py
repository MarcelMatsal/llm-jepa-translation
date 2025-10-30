"""
Evaluation metrics for embedding alignment.
"""
import torch
import torch.nn.functional as F


def compute_metrics(model, test_loader, device='cuda'):
    """
    Compute evaluation metrics.
    
    Returns:
        Dictionary of metrics including:
        - cosine_similarity: Average cosine similarity between aligned pairs
        - mse: Mean squared error
        - embedding_diversity: Measure of embedding diversity (prevent collapse)
        - linearity_error: Least-squares error (measure of linearity)
    """
    model.eval()
    all_s_x = []
    all_s_y = []
    all_s_y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            texts_src = batch['texts_src']
            texts_tgt = batch['texts_tgt']
            lang_src = torch.tensor(batch['lang_src']).to(device)
            lang_tgt = torch.tensor(batch['lang_tgt']).to(device)
            
            s_x, s_y, s_y_pred = model.forward(
                texts_src, texts_tgt, lang_src, lang_tgt, normalize=True
            )
            
            all_s_x.append(s_x.cpu())
            all_s_y.append(s_y.cpu())
            all_s_y_pred.append(s_y_pred.cpu())
    
    # Concatenate all embeddings
    s_x = torch.cat(all_s_x, dim=0)
    s_y = torch.cat(all_s_y, dim=0)
    s_y_pred = torch.cat(all_s_y_pred, dim=0)
    
    # Cosine similarity
    cosine_sim = F.cosine_similarity(s_y_pred, s_y, dim=-1).mean().item()
    
    # MSE
    mse = F.mse_loss(s_y_pred, s_y).item()
    
    # Embedding diversity (prevent collapse)
    # Measure: std of embeddings (higher = more diverse)
    embedding_std = s_x.std(dim=0).mean().item()
    
    # Linearity check: least-squares error
    # Check if predictor learns approximately linear transformation
    try:
        W_opt, _ = torch.lstsq(s_x, s_y)
        linearity_error = torch.norm(s_y - s_x @ W_opt).item()
    except:
        linearity_error = float('inf')
    
    # Singular values of difference (they use this in llm jepa, can talk to randall about this)
    diff = s_x - s_y
    U, S, V = torch.svd(diff)
    top_singular_values = S[:10].tolist()
    
    return {
        'cosine_similarity': cosine_sim,
        'mse': mse,
        'embedding_diversity': embedding_std,
        'linearity_error': linearity_error,
        'top_singular_values': top_singular_values
    }

