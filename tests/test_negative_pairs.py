"""
Test negative pair cosine similarity computation logic.
"""
import torch
import torch.nn.functional as F

def test_negative_pair_similarity():
    """Test the negative pair similarity computation."""
    # Create sample embeddings
    batch_size = 10
    d_model = 768
    
    # Create random normalized embeddings
    cls1 = F.normalize(torch.randn(batch_size, d_model), p=2, dim=-1)
    cls2 = F.normalize(torch.randn(batch_size, d_model), p=2, dim=-1)
    
    # Compute positive pair similarities (diagonal)
    positive_sims = F.cosine_similarity(cls1, cls2, dim=-1)
    positive_sim_mean = positive_sims.mean().item()
    
    print(f"Positive pair similarity: {positive_sim_mean:.4f}")
    
    # Compute all pairwise similarities
    similarity_matrix = torch.mm(cls1, cls2.t())
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # Extract negative pairs (off-diagonal)
    mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool)
    negative_sims = similarity_matrix[~mask]
    
    print(f"Number of negative pairs: {negative_sims.shape[0]}")
    print(f"Expected negative pairs: {batch_size * (batch_size - 1)}")
    
    # Compute statistics
    negative_sim_mean = negative_sims.mean().item()
    negative_sim_std = negative_sims.std().item()
    negative_sim_median = negative_sims.median().item()
    
    print(f"\nNegative pair statistics:")
    print(f"  Mean: {negative_sim_mean:.4f}")
    print(f"  Std: {negative_sim_std:.4f}")
    print(f"  Median: {negative_sim_median:.4f}")
    
    # Compute gap and ratio
    sim_gap = positive_sim_mean - negative_sim_mean
    sim_ratio = positive_sim_mean / (negative_sim_mean + 1e-8)
    
    print(f"\nDiscrimination metrics:")
    print(f"  Gap (positive - negative): {sim_gap:.4f}")
    print(f"  Ratio (positive / negative): {sim_ratio:.4f}")
    
    # Verify the diagonal of similarity matrix matches positive similarities
    diagonal_sims = torch.diagonal(similarity_matrix)
    assert torch.allclose(diagonal_sims, positive_sims, atol=1e-6), "Diagonal should match positive pairs"
    
    print("\n✓ Test passed!")

if __name__ == "__main__":
    test_negative_pair_similarity()
