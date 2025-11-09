"""
Training loop for dual-objective BERT model.
Combines MLM loss and CLS alignment loss.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional, List
import os
import json


class DualObjectiveTrainer:
    """
    Trainer for BertDualObjective model.
    Handles training with combined MLM + alignment loss.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        max_grad_norm: float = 1.0,
        log_interval: int = 100,
        save_dir: str = './checkpoints',
        accumulation_steps: int = 1
    ):
        """
        Args:
            model: BertDualObjective model
            train_loader: Training data loader (with DualObjectiveCollator)
            val_loader: Validation data loader (optional)
            optimizer: Optimizer (default: AdamW)
            scheduler: Learning rate scheduler (optional)
            device: 'cuda' or 'cpu'
            max_grad_norm: Gradient clipping norm
            log_interval: Logging frequency (steps)
            save_dir: Directory to save checkpoints
            accumulation_steps: Gradient accumulation steps
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.accumulation_steps = accumulation_steps
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=2e-5,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.history = []
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with average training metrics
        """
        self.model.train()
        
        total_metrics = {}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Compute loss
            loss, metrics = self.model.compute_total_loss(batch)
            
            # Normalize loss for gradient accumulation
            loss = loss / self.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Accumulate metrics
            for key, value in metrics.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value
            num_batches += 1
            
            # Update weights (every accumulation_steps)
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.log_interval == 0:
                    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{avg_metrics['total_loss']:.4f}",
                        'mlm': f"{avg_metrics['mlm_loss']:.4f}",
                        'align': f"{avg_metrics['alignment_loss']:.4f}",
                        'cos_sim': f"{avg_metrics['cls_cosine_sim']:.4f}"
                    })
                    
                    # Reset metrics for next interval
                    total_metrics = {}
                    num_batches = 0
        
        # Handle remaining gradients
        if len(self.train_loader) % self.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
        
        # Compute average metrics for the epoch
        avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}
        
        return avg_metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary with average validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Compute loss
                loss, metrics = self.model.compute_total_loss(batch)
                
                # Accumulate metrics
                for key, value in metrics.items():
                    total_metrics[key] = total_metrics.get(key, 0.0) + value
                num_batches += 1
        
        # Compute averages
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return avg_metrics
    
    def train(self, num_epochs: int, save_every: int = 1):
        """
        Train for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Total steps per epoch: {len(self.train_loader)}")
        print(f"Device: {self.device}")
        print(f"Gradient accumulation steps: {self.accumulation_steps}")
        print(f"Effective batch size: {self.train_loader.batch_size * self.accumulation_steps}")
        print()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_metrics.get('total_loss', 0.0):.4f} "
                  f"(MLM: {train_metrics.get('mlm_loss', 0.0):.4f}, "
                  f"Align: {train_metrics.get('alignment_loss', 0.0):.4f})")
            print(f"  Train CLS Similarity: {train_metrics.get('cls_cosine_sim', 0.0):.4f}")
            print(f"  Train MLM Accuracy: {train_metrics.get('mlm_accuracy', 0.0):.4f}")
            
            if val_metrics:
                print(f"  Val Loss: {val_metrics.get('total_loss', 0.0):.4f} "
                      f"(MLM: {val_metrics.get('mlm_loss', 0.0):.4f}, "
                      f"Align: {val_metrics.get('alignment_loss', 0.0):.4f})")
                print(f"  Val CLS Similarity: {val_metrics.get('cls_cosine_sim', 0.0):.4f}")
            
            # Save history
            epoch_history = {
                'epoch': epoch + 1,
                'global_step': self.global_step,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            self.history.append(epoch_history)
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
            
            # Save best model
            if val_metrics and val_metrics.get('total_loss', float('inf')) < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.save_checkpoint('best_model.pt')
                print(f"  ✓ New best model saved (val_loss: {self.best_val_loss:.4f})")
            
            print()
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        
        # Save training history
        self.save_history()
        
        print("Training complete!")
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        # Also save model in HuggingFace format
        model_dir = os.path.join(self.save_dir, filename.replace('.pt', ''))
        os.makedirs(model_dir, exist_ok=True)
        self.model.save_pretrained(model_dir)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint.get('history', [])
        
        print(f"Loaded checkpoint from epoch {self.epoch}, step {self.global_step}")
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")
