"""
InfoNCE: Information Noise-Contrastive Estimation

Based on:
- "Representation Learning with Contrastive Predictive Coding" (van den Oord et al., 2018)
- SimCLR (Chen et al., 2020)
- SSL Cookbook recommendations

InfoNCE prevents collapse by using negative pairs:
- Positive: (CLS_lang1, CLS_lang2) from same translation
- Negatives: All other CLS tokens in batch (both languages)

This adds the missing "repulsion" term that prevents all embeddings
from collapsing to the same point.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

from ..base import FunctionalLoss, ArchitecturalLoss
from ..registry import register_loss


@register_loss('infonce')
class InfoNCELoss(ArchitecturalLoss):
    """
    InfoNCE loss for contrastive learning.
    
    Uses in-batch negatives: all other CLS tokens serve as negative examples.
    
    Args:
        config: Configuration dictionary with keys:
            - temperature: Temperature scaling parameter (default: 0.07)
            - normalize: Whether to L2 normalize embeddings (default: True)
            - use_projector: Whether to use MLP projector (default: False)
            - projector_hidden_dim: Hidden dimension for projector (default: 2048)
            - projector_output_dim: Output dimension for projector (default: 128)
    
    Example:
        >>> config = {
        ...     'type': 'infonce',
        ...     'temperature': 0.07,
        ...     'normalize': True,
        ...     'use_projector': False
        ... }
        >>> loss_fn = InfoNCELoss(config, components={})
        >>> z1 = torch.randn(32, 768)  # CLS tokens lang1
        >>> z2 = torch.randn(32, 768)  # CLS tokens lang2
        >>> loss, metrics = loss_fn.compute(z1, z2)
    """
    
    def __init__(self, config: Dict, components: Dict):
        super().__init__(config, components)
        self.temperature = config.get('temperature', 0.07)
        self.normalize = config.get('normalize', True)
        self.use_projector = config.get('use_projector', False)
    
    @staticmethod
    def get_required_components() -> Dict[str, type]:
        """
        InfoNCE can optionally use a projector.
        The factory will create it if use_projector=True in config.
        """
        # We'll handle this dynamically based on config
        return {}
    
    def compute(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute InfoNCE loss with in-batch negatives.
        
        Args:
            z1: CLS embeddings from language 1 (batch_size, dim)
            z2: CLS embeddings from language 2 (batch_size, dim)
        
        Returns:
            loss: InfoNCE loss (scalar)
            metrics: Dictionary with loss and similarity statistics
        """
        # Apply projector if it exists
        if self.use_projector and 'projector' in self.components:
            z1 = self.components['projector'](z1)
            z2 = self.components['projector'](z2)
        
        # L2 normalize embeddings (standard for cosine similarity)
        if self.normalize:
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)
        
        batch_size = z1.size(0)
        device = z1.device
        
        # CRITICAL FIX: Concatenate z1 and z2 to use ALL CLS tokens as negatives
        # This provides within-language negatives (z1[i] vs z1[j]) 
        # AND cross-language negatives (z1[i] vs z2[j])
        # Shape: (2*batch_size, dim)
        z_all = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix between z1 and ALL tokens
        # Shape: (batch_size, 2*batch_size)
        logits_1 = (z1 @ z_all.T) / self.temperature
        
        # Compute similarity matrix between z2 and ALL tokens
        # Shape: (batch_size, 2*batch_size)
        logits_2 = (z2 @ z_all.T) / self.temperature
        
        # Labels: For z1[i], the positive is z_all[i + batch_size] = z2[i]
        # For z2[i], the positive is z_all[i] = z1[i]
        labels_1 = torch.arange(batch_size, 2 * batch_size, device=device)
        labels_2 = torch.arange(batch_size, device=device)
        
        # Contrastive loss
        # z1[i] should match z2[i] (at position batch_size + i in z_all)
        loss_1 = F.cross_entropy(logits_1, labels_1)
        # z2[i] should match z1[i] (at position i in z_all)
        loss_2 = F.cross_entropy(logits_2, labels_2)
        
        # Average both directions
        loss = (loss_1 + loss_2) / 2
        
        # Compute metrics
        with torch.no_grad():
            # Positive pair similarities
            # For z1: positive is z2 (columns batch_size:2*batch_size)
            pos_sim_1 = torch.diagonal(logits_1[:, batch_size:]).mean() * self.temperature
            # For z2: positive is z1 (columns 0:batch_size)
            pos_sim_2 = torch.diagonal(logits_2[:, :batch_size]).mean() * self.temperature
            pos_sim = (pos_sim_1 + pos_sim_2) / 2
            
            # Negative pair similarities (all non-diagonal)
            # Create masks for positive pairs
            mask_1 = torch.ones(batch_size, 2 * batch_size, dtype=torch.bool, device=device)
            mask_1[torch.arange(batch_size), batch_size + torch.arange(batch_size)] = False
            
            mask_2 = torch.ones(batch_size, 2 * batch_size, dtype=torch.bool, device=device)
            mask_2[torch.arange(batch_size), torch.arange(batch_size)] = False
            
            neg_sim_1 = logits_1[mask_1].mean() * self.temperature
            neg_sim_2 = logits_2[mask_2].mean() * self.temperature
            neg_sim = (neg_sim_1 + neg_sim_2) / 2
            
            # Accuracy: how often is positive the highest similarity?
            pred_1 = logits_1.argmax(dim=1)
            pred_2 = logits_2.argmax(dim=1)
            acc_1 = (pred_1 == labels_1).float().mean()
            acc_2 = (pred_2 == labels_2).float().mean()
            accuracy = (acc_1 + acc_2) / 2
            
            # Compute within-language vs cross-language negative similarities
            # Within-language negatives for z1: z1[i] vs z1[j] (j != i)
            within_lang_mask_1 = torch.ones(batch_size, batch_size, dtype=torch.bool, device=device)
            within_lang_mask_1.fill_diagonal_(False)
            within_lang_sim = (z1 @ z1.T)[within_lang_mask_1].mean()
            
            # Cross-language negatives (excluding positives): z1[i] vs z2[j] (j != i)
            cross_lang_sim_matrix = z1 @ z2.T
            cross_lang_mask = torch.ones(batch_size, batch_size, dtype=torch.bool, device=device)
            cross_lang_mask.fill_diagonal_(False)
            cross_lang_neg_sim = cross_lang_sim_matrix[cross_lang_mask].mean()
        
        metrics = {
            'alignment_loss': loss.item(),
            'infonce_loss': loss.item(),
            'loss_1': loss_1.item(),
            'loss_2': loss_2.item(),
            'positive_sim': pos_sim.item(),
            'negative_sim': neg_sim.item(),
            'within_lang_neg_sim': within_lang_sim.item(),
            'cross_lang_neg_sim': cross_lang_neg_sim.item(),
            'contrastive_accuracy': accuracy.item(),
            'temperature': self.temperature,
            'batch_size': batch_size,
            'num_negatives': 2 * batch_size - 2  # Exclude anchor itself AND its positive
        }
        
        return loss, metrics
