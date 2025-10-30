"""
Main JEPA model: Separate encoders with EMA updates and language-conditioned predictor.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import SentenceEncoder
from .predictor import Predictor


class MultilingualJEPA(nn.Module):
    """
    Multilingual JEPA for learning aligned embeddings across languages.
    
    Architecture:
    - X-Encoder (online): Gets gradients
    - Y-Encoder (target): Updated via EMA, no gradients from loss
    - Predictor: Language-conditioned predictor shared across all language pairs
    """
    
    def __init__(
        self,
        encoder_name='bert-base-multilingual-cased',
        pooling='cls',
        d_model=None,
        num_languages=2,
        lang_embed_dim=None,
        tau=0.999
    ):
        """
        Args:
            encoder_name: HuggingFace model identifier for encoders
            pooling: 'cls', 'mean', or 'attention'
            d_model: Embedding dimension (auto-detected from encoder)
            num_languages: Number of languages (for language embeddings)
            lang_embed_dim: Dimension of language embeddings (default: d_model)
            tau: EMA decay rate (0.999 = slow updates)
        """
        super().__init__()
        self.tau = tau
        
        # Separate encoders
        self.x_encoder = SentenceEncoder(encoder_name, pooling=pooling)
        self.y_encoder = SentenceEncoder(encoder_name, pooling=pooling)
        
        # Initialize y_encoder from x_encoder
        self.y_encoder.load_state_dict(self.x_encoder.state_dict())
        
        # Freeze y_encoder (will be updated via EMA only)
        for param in self.y_encoder.parameters():
            param.requires_grad = False
        
        # Get embedding dimension
        self.d_model = d_model or self.x_encoder.d_model
        self.lang_embed_dim = lang_embed_dim or self.d_model
        
        # Language embeddings
        self.lang_embedding = nn.Embedding(num_languages, self.lang_embed_dim)
        
        # Predictor (shared across all language pairs)
        self.predictor = Predictor(
            d_model=self.d_model,
            lang_embed_dim=self.lang_embed_dim,
            hidden_dim=2 * self.d_model
        )
    
    def forward(self, texts_x, texts_y, lang_x, lang_y, normalize=True):
        """
        Forward pass: encode and predict.
        
        Args:
            texts_x: Source language texts (list of strings)
            texts_y: Target language texts (list of strings)
            lang_x: Source language IDs (tensor of shape batch_size)
            lang_y: Target language IDs (tensor of shape batch_size)
            normalize: Whether to normalize embeddings before loss
        
        Returns:
            s_x: Source embeddings (batch_size, d_model)
            s_y: Target embeddings (batch_size, d_model) [detached]
            s_y_pred: Predicted target embeddings (batch_size, d_model)
        """
        # Encode both languages
        s_x = self.x_encoder(texts_x)  # (batch, d_model)
        s_y = self.y_encoder(texts_y)  # (batch, d_model)
        
        # Detach y-encoder output (no gradients)
        s_y = s_y.detach()
        
        # Normalize embeddings
        if normalize:
            s_x = F.normalize(s_x, p=2, dim=-1)
            s_y = F.normalize(s_y, p=2, dim=-1)
        
        # Get language embeddings
        lang_emb_x = self.lang_embedding(lang_x)  # (batch, lang_embed_dim)
        lang_emb_y = self.lang_embedding(lang_y)  # (batch, lang_embed_dim)
        
        # Predict target embedding
        s_y_pred = self.predictor(s_x, lang_emb_x, lang_emb_y)
        
        # Normalize prediction
        if normalize:
            s_y_pred = F.normalize(s_y_pred, p=2, dim=-1)
        
        return s_x, s_y, s_y_pred
    
    def compute_loss(self, texts_x, texts_y, lang_x, lang_y, loss_type='mse'):
        """
        Compute JEPA loss (bidirectional).
        
        Args:
            texts_x: Source language texts
            texts_y: Target language texts
            lang_x: Source language IDs
            lang_y: Target language IDs
            loss_type: 'mse' or 'cosine'
        
        Returns:
            loss: Total bidirectional loss
            metrics: Dictionary of loss components
        """
        # Forward direction: x → y
        s_x, s_y, s_y_pred = self.forward(texts_x, texts_y, lang_x, lang_y)
        
        if loss_type == 'mse':
            loss_forward = F.mse_loss(s_y_pred, s_y)
        elif loss_type == 'cosine':
            loss_forward = 1 - F.cosine_similarity(s_y_pred, s_y).mean()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Backward direction: y → x
        s_y_rev, s_x_rev, s_x_pred = self.forward(texts_y, texts_x, lang_y, lang_x)
        
        if loss_type == 'mse':
            loss_backward = F.mse_loss(s_x_pred, s_x_rev)
        elif loss_type == 'cosine':
            loss_backward = 1 - F.cosine_similarity(s_x_pred, s_x_rev).mean()
        
        # Total loss
        loss = loss_forward + loss_backward
        
        metrics = {
            'loss': loss.item(),
            'loss_forward': loss_forward.item(),
            'loss_backward': loss_backward.item(),
            'cosine_sim_forward': F.cosine_similarity(s_y_pred, s_y).mean().item(),
            'cosine_sim_backward': F.cosine_similarity(s_x_pred, s_x_rev).mean().item()
        }
        
        return loss, metrics
    
    def update_ema(self):
        """Update y-encoder via EMA of x-encoder weights."""
        with torch.no_grad():
            for param_x, param_y in zip(
                self.x_encoder.parameters(),
                self.y_encoder.parameters()
            ):
                param_y.data = self.tau * param_y.data + (1 - self.tau) * param_x.data
    
    def get_embeddings(self, texts, encoder='x'):
        """
        Get embeddings for texts (for evaluation/inference).
        
        Args:
            texts: List of strings
            encoder: 'x' or 'y'
        
        Returns:
            embeddings: (batch_size, d_model)
        """
        encoder_model = self.x_encoder if encoder == 'x' else self.y_encoder
        encoder_model.eval()
        with torch.no_grad():
            embeddings = encoder_model(texts)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

