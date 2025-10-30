"""
Training loop with EMA updates and bidirectional loss.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional


class Trainer:
    """Training loop for Multilingual JEPA."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cuda',
        max_grad_norm: float = 1.0,
        log_interval: int = 100
    ):
        """
        Args:
            model: MultilingualJEPA model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            optimizer: Optimizer (default: Adam)
            device: 'cuda' or 'cpu'
            max_grad_norm: Gradient clipping norm
            log_interval: Logging frequency
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        
        if optimizer is None:
            # Only optimize x_encoder and predictor (y_encoder updated via EMA)
            trainable_params = list(self.model.x_encoder.parameters()) + \
                             list(self.model.predictor.parameters()) + \
                             list(self.model.lang_embedding.parameters())
            self.optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
        else:
            self.optimizer = optimizer
        
        self.step = 0
        self.history = []
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_metrics = {}
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Move to device
            texts_src = batch['texts_src']
            texts_tgt = batch['texts_tgt']
            lang_src = torch.tensor(batch['lang_src']).to(self.device)
            lang_tgt = torch.tensor(batch['lang_tgt']).to(self.device)
            
            # Forward and loss
            loss, metrics = self.model.compute_loss(
                texts_src, texts_tgt, lang_src, lang_tgt, loss_type='mse'
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # EMA update
            self.model.update_ema()
            
            # Logging
            total_loss += loss.item()
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            
            self.step += 1
            
            if self.step % self.log_interval == 0:
                avg_loss = total_loss / self.log_interval
                avg_metrics = {k: v / self.log_interval for k, v in total_metrics.items()}
                pbar.set_postfix({'loss': avg_loss, **avg_metrics})
                total_loss = 0.0
                total_metrics = {}
        
        return avg_loss, avg_metrics
    
    def validate(self):
        """Validate on validation set."""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0.0
        total_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                texts_src = batch['texts_src']
                texts_tgt = batch['texts_tgt']
                lang_src = torch.tensor(batch['lang_src']).to(self.device)
                lang_tgt = torch.tensor(batch['lang_tgt']).to(self.device)
                
                loss, metrics = self.model.compute_loss(
                    texts_src, texts_tgt, lang_src, lang_tgt, loss_type='mse'
                )
                
                total_loss += loss.item()
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v
        
        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = {k: v / len(self.val_loader) for k, v in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def train(self, num_epochs: int):
        """Train for multiple epochs."""
        for epoch in range(1, num_epochs + 1):
            print(f'\nEpoch {epoch}/{num_epochs}')
            train_loss, train_metrics = self.train_epoch()
            
            if self.val_loader:
                val_loss, val_metrics = self.validate()
                print(f'Val Loss: {val_loss:.4f}')
            
            self.history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_metrics': train_metrics
            })

