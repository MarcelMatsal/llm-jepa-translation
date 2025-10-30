"""
Language-conditioned predictor network.
Predicts target language embedding from source language embedding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    """Predictor that maps source embedding to target embedding conditioned on language pair."""
    
    def __init__(self, d_model, lang_embed_dim=None, hidden_dim=None):
        """
        Args:
            d_model: Dimension of sentence embeddings
            lang_embed_dim: Dimension of language embeddings (default: d_model)
            hidden_dim: Hidden dimension of predictor MLP (default: 2 * d_model)
        """
        super().__init__()
        self.d_model = d_model
        self.lang_embed_dim = lang_embed_dim or d_model
        self.hidden_dim = hidden_dim or (2 * d_model)
        
        # Predictor: MLP that takes s_x + language conditioning
        # Input: s_x (d_model) + lang_emb_src (lang_embed_dim) + lang_emb_tgt (lang_embed_dim)
        input_dim = d_model + 2 * self.lang_embed_dim
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, d_model)
        )
    
    def forward(self, s_x, lang_emb_src, lang_emb_tgt):
        """
        Predict target embedding from source embedding.
        
        Args:
            s_x: Source embeddings (batch_size, d_model)
            lang_emb_src: Source language embeddings (batch_size, lang_embed_dim)
            lang_emb_tgt: Target language embeddings (batch_size, lang_embed_dim)
        
        Returns:
            s_y_pred: Predicted target embeddings (batch_size, d_model)
        """
        # Concatenate source embedding with language conditioning
        z = torch.cat([lang_emb_src, lang_emb_tgt], dim=-1)  # (batch, 2 * lang_embed_dim)
        combined = torch.cat([s_x, z], dim=-1)  # (batch, d_model + 2 * lang_embed_dim)
        
        # Predict target embedding
        s_y_pred = self.predictor(combined)
        
        return s_y_pred

