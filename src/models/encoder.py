"""
Sentence encoder using Transformer architecture.
Extracts CLS token or uses pooling for variable-length sequences.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class SentenceEncoder(nn.Module):
    """Transformer encoder that outputs fixed-size sentence embeddings."""
    
    def __init__(self, model_name='bert-base-multilingual-cased', pooling='cls'):
        """
        Args:
            model_name: HuggingFace model identifier
            pooling: 'cls' (use CLS token), 'mean' (mean pooling), 'attention' (learned attention)
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling = pooling
        self.d_model = self.model.config.hidden_size
        
        # Add CLS token if not present
        if self.tokenizer.cls_token is None:
            self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Learned attention pooling
        if pooling == 'attention':
            self.attention_pool = nn.Linear(self.d_model, 1)
    
    def forward(self, texts, return_tokens=False):
        """
        Args:
            texts: List of strings or batched tokenized input
            return_tokens: If True, return tokenized inputs
        
        Returns:
            embeddings: (batch_size, d_model) sentence embeddings
            (optional) tokenized: tokenized inputs if return_tokens=True
        """
        # Tokenize if needed
        if isinstance(texts, list):
            tokenized = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
        else:
            tokenized = texts
        
        # Move to device
        device = next(self.model.parameters()).device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        # Forward pass
        outputs = self.model(**tokenized)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, d_model)
        
        # Extract embeddings
        if self.pooling == 'cls':
            # Use CLS token (position 0)
            embeddings = hidden_states[:, 0, :]  # (batch, d_model)
        elif self.pooling == 'mean':
            # Mean pooling (mask out padding)
            attention_mask = tokenized['attention_mask'].unsqueeze(-1)  # (batch, seq_len, 1)
            masked_hidden = hidden_states * attention_mask
            embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)  # (batch, d_model)
        elif self.pooling == 'attention':
            # Learned attention pooling
            attention_mask = tokenized['attention_mask']
            weights = self.attention_pool(hidden_states)  # (batch, seq_len, 1)
            weights = weights.masked_fill(attention_mask.unsqueeze(-1) == 0, float('-inf'))
            weights = torch.softmax(weights, dim=1)
            embeddings = (hidden_states * weights).sum(dim=1)  # (batch, d_model)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        if return_tokens:
            return embeddings, tokenized
        return embeddings
    
    def encode(self, texts):
        """Convenience method for encoding."""
        self.eval()
        with torch.no_grad():
            return self.forward(texts)

