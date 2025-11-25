"""
Training loop for dual-objective BERT model.
Combines MLM loss and CLS alignment loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Dict, Optional, List
import os
import json
import shutil
import wandb
from huggingface_hub import HfApi, create_repo


def compute_rankme(embeddings: torch.Tensor) -> float:
    """
    Compute RankMe (effective rank) of embeddings.
    
    RankMe is an unsupervised metric for assessing the quality of self-supervised
    representations by measuring their effective rank. Higher rank indicates better
    utilization of the embedding space, while low rank suggests representation collapse.
    
    Reference: "RankMe: Assessing the downstream performance of pretrained 
    self-supervised representations by their rank" (arXiv:2210.02885)
    
    Args:
        embeddings: (N, d) tensor of embeddings where N is number of samples
                   and d is embedding dimension
    
    Returns:
        Effective rank (scalar). Maximum possible value is min(N, d).
        Higher values indicate better representation quality.
    """
    # Compute SVD to get singular values
    # We only need singular values, not U and V matrices
    try:
        _, S, _ = torch.svd(embeddings)
    except RuntimeError:
        # Fallback if SVD fails (e.g., for very large matrices)
        # Use a more stable but slower method
        S = torch.linalg.svdvals(embeddings)
    
    # Normalize singular values to get a probability distribution
    p = (S / S.sum()) + 1e-10  # Add small epsilon for numerical stability
    
    # Compute Shannon entropy of the singular value distribution
    entropy = -(p * torch.log(p)).sum()
    
    # Effective rank is exp(entropy)
    # This gives a measure between 1 (completely collapsed) and min(N, d) (full rank)
    return torch.exp(entropy).item()


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
        accumulation_steps: int = 1,
        use_wandb: bool = True,
        use_amp: bool = True,
        use_gradient_checkpointing: bool = True,
        tokenizer = None,
        hub_model_id: Optional[str] = None,
        push_to_hub: bool = False,
        experiment_metadata: Optional[Dict] = None
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
            save_dir: Directory to save checkpoints (local)
            accumulation_steps: Gradient accumulation steps
            use_wandb: Whether to log to Weights & Biases
            use_amp: Use automatic mixed precision (FP16) for memory efficiency
            use_gradient_checkpointing: Enable gradient checkpointing to save memory
            tokenizer: Tokenizer to save with model checkpoints (optional but recommended)
            hub_model_id: HuggingFace Hub model ID (e.g., 'username/model-name')
            push_to_hub: Whether to push checkpoints to HuggingFace Hub
            experiment_metadata: Metadata about the experiment (for model card and commit messages)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        # Convert to absolute path to avoid issues with relative paths
        self.save_dir = os.path.abspath(save_dir)
        self.accumulation_steps = accumulation_steps
        self.use_wandb = use_wandb
        self.use_amp = use_amp
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.tokenizer = tokenizer
        self.hub_model_id = hub_model_id
        self.push_to_hub = push_to_hub
        self.experiment_metadata = experiment_metadata or {}
        
        # Enable gradient checkpointing if requested
        if self.use_gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("✓ Gradient checkpointing enabled")
            elif hasattr(self.model.mlm_model, 'gradient_checkpointing_enable'):
                self.model.mlm_model.gradient_checkpointing_enable()
                print("✓ Gradient checkpointing enabled on MLM model")
        
        # Initialize GradScaler for mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("✓ Automatic Mixed Precision (AMP) enabled")
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Create HuggingFace Hub repository if pushing to hub
        if self.push_to_hub and self.hub_model_id:
            try:
                create_repo(self.hub_model_id, exist_ok=True, repo_type="model")
                print(f"✓ HuggingFace Hub repository ready: {self.hub_model_id}")
            except Exception as e:
                print(f"⚠ Warning: Could not create/verify Hub repository: {e}")
                print(f"  Make sure you're logged in with 'huggingface-cli login'")
        
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
        
        # Watch model with wandb for gradient and parameter tracking
        if self.use_wandb and wandb.run is not None:
            wandb.watch(self.model, log='all', log_freq=self.log_interval)
    
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
            
            # Mixed precision forward pass
            if self.use_amp:
                with autocast():
                    loss, metrics = self.model.compute_total_loss(batch)
                    # Normalize loss for gradient accumulation
                    loss = loss / self.accumulation_steps
            else:
                loss, metrics = self.model.compute_total_loss(batch)
                # Normalize loss for gradient accumulation
                loss = loss / self.accumulation_steps
            
            # Backward pass with gradient scaling
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulate metrics (detach to avoid keeping computation graph)
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach().item()
                total_metrics[key] = total_metrics.get(key, 0.0) + value
            num_batches += 1
            
            # Free memory from batch
            del batch, loss, metrics
            
            # Clear CUDA cache periodically to prevent fragmentation
            if (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
            
            # Update weights (every accumulation_steps)
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping (unscale first if using AMP)
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                # Optimizer step with gradient scaling
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Zero gradients (set_to_none=True for better memory efficiency)
                self.optimizer.zero_grad(set_to_none=True)
                
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
                    
                    # Log to wandb
                    if self.use_wandb and wandb.run is not None:
                        log_dict = {f'train/{k}': v for k, v in avg_metrics.items()}
                        log_dict['train/learning_rate'] = self.optimizer.param_groups[0]['lr']
                        log_dict['train/global_step'] = self.global_step
                        log_dict['train/epoch'] = self.epoch
                        wandb.log(log_dict, step=self.global_step)
                    
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
        
        # Accumulate CLS embeddings for negative pair analysis
        all_cls1 = []
        all_cls2 = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Compute loss and get CLS embeddings
                # We need to extract CLS embeddings separately for negative pair analysis
                cls1 = self.model.extract_cls_embeddings(
                    batch['cls1_input_ids'],
                    batch['cls1_attention_mask'],
                    batch['cls1_positions']
                )
                cls2 = self.model.extract_cls_embeddings(
                    batch['cls2_input_ids'],
                    batch['cls2_attention_mask'],
                    batch['cls2_positions']
                )
                
                # Normalize embeddings
                cls1_norm = F.normalize(cls1, p=2, dim=-1)
                cls2_norm = F.normalize(cls2, p=2, dim=-1)
                
                # Store for negative pair computation
                all_cls1.append(cls1_norm.cpu())
                all_cls2.append(cls2_norm.cpu())
                
                # Compute loss using the model's method
                loss, metrics = self.model.compute_total_loss(batch)
                
                # Accumulate metrics
                for key, value in metrics.items():
                    total_metrics[key] = total_metrics.get(key, 0.0) + value
                num_batches += 1
        
        # Compute averages
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        # Compute negative pair similarities
        # Concatenate all embeddings
        all_cls1 = torch.cat(all_cls1, dim=0)  # (N, d_model)
        all_cls2 = torch.cat(all_cls2, dim=0)  # (N, d_model)
        
        N = all_cls1.shape[0]
        d_model = all_cls1.shape[1]
        
        # Compute RankMe for both CLS token sets
        # RankMe measures the effective rank of representations
        # Higher values (closer to d_model) indicate better representation quality
        # Low values suggest representation collapse
        rankme_cls1 = compute_rankme(all_cls1)
        rankme_cls2 = compute_rankme(all_cls2)
        
        # Add RankMe metrics
        avg_metrics['rankme_cls1'] = rankme_cls1
        avg_metrics['rankme_cls2'] = rankme_cls2
        avg_metrics['rankme_mean'] = (rankme_cls1 + rankme_cls2) / 2
        # Normalized RankMe (as percentage of maximum possible rank)
        max_rank = min(N, d_model)
        avg_metrics['rankme_cls1_normalized'] = rankme_cls1 / max_rank
        avg_metrics['rankme_cls2_normalized'] = rankme_cls2 / max_rank
        avg_metrics['rankme_mean_normalized'] = ((rankme_cls1 + rankme_cls2) / 2) / max_rank
        
        # Compute positive pair similarity (already in avg_metrics as cls_cosine_sim)
        positive_sim = avg_metrics.get('cls_cosine_sim', 0.0)
        
        # Compute negative pair similarities
        # For efficiency, we'll sample if N is very large
        max_samples = min(N, 1000)  # Limit to avoid memory issues
        
        if N > max_samples:
            # Randomly sample indices
            import random
            indices = random.sample(range(N), max_samples)
            cls1_sample = all_cls1[indices]
            cls2_sample = all_cls2[indices]
        else:
            cls1_sample = all_cls1
            cls2_sample = all_cls2
        
        # Compute all pairwise similarities: cls1[i] · cls2[j]
        # Shape: (max_samples, max_samples)
        similarity_matrix = torch.mm(cls1_sample, cls2_sample.t())
        
        # Extract negative pairs (off-diagonal elements)
        # Create mask for diagonal elements
        mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool)
        negative_sims = similarity_matrix[~mask]
        
        # Compute statistics for negative pairs
        negative_sim_mean = negative_sims.mean().item()
        negative_sim_std = negative_sims.std().item()
        negative_sim_median = negative_sims.median().item()
        
        # Compute gap and ratio
        sim_gap = positive_sim - negative_sim_mean
        sim_ratio = positive_sim / (negative_sim_mean + 1e-8)  # Add epsilon to avoid division by zero
        
        # Add negative pair metrics
        avg_metrics['cls_cosine_sim_positive'] = positive_sim
        avg_metrics['cls_cosine_sim_negative'] = negative_sim_mean
        avg_metrics['cls_cosine_sim_negative_std'] = negative_sim_std
        avg_metrics['cls_cosine_sim_negative_median'] = negative_sim_median
        avg_metrics['cls_cosine_sim_gap'] = sim_gap
        avg_metrics['cls_cosine_sim_ratio'] = sim_ratio
        
        # Log to wandb
        if self.use_wandb and wandb.run is not None:
            log_dict = {f'val/{k}': v for k, v in avg_metrics.items()}
            log_dict['val/epoch'] = self.epoch
            wandb.log(log_dict, step=self.global_step)
        
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
        
        # Log training config to wandb
        if self.use_wandb and wandb.run is not None:
            wandb.config.update({
                'num_epochs': num_epochs,
                'save_every': save_every,
                'steps_per_epoch': len(self.train_loader),
                'accumulation_steps': self.accumulation_steps,
                'effective_batch_size': self.train_loader.batch_size * self.accumulation_steps,
                'max_grad_norm': self.max_grad_norm,
                'device': str(self.device)
            }, allow_val_change=True)
        
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
                print(f"  Val CLS Similarity - Positive: {val_metrics.get('cls_cosine_sim_positive', 0.0):.4f}, "
                      f"Negative: {val_metrics.get('cls_cosine_sim_negative', 0.0):.4f}, "
                      f"Gap: {val_metrics.get('cls_cosine_sim_gap', 0.0):.4f}")
                print(f"  Val RankMe - CLS1: {val_metrics.get('rankme_cls1', 0.0):.2f}, "
                      f"CLS2: {val_metrics.get('rankme_cls2', 0.0):.2f}, "
                      f"Mean: {val_metrics.get('rankme_mean', 0.0):.2f} "
                      f"({val_metrics.get('rankme_mean_normalized', 0.0):.1%} of max)")
            
            
            # Save history
            epoch_history = {
                'epoch': epoch + 1,
                'global_step': self.global_step,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            self.history.append(epoch_history)
            
            # Log epoch summary to wandb
            if self.use_wandb and wandb.run is not None:
                epoch_summary = {
                    'epoch/number': epoch + 1,
                    'epoch/train_loss': train_metrics.get('total_loss', 0.0),
                    'epoch/train_mlm_loss': train_metrics.get('mlm_loss', 0.0),
                    'epoch/train_alignment_loss': train_metrics.get('alignment_loss', 0.0),
                    'epoch/train_cls_similarity': train_metrics.get('cls_cosine_sim', 0.0),
                    'epoch/train_mlm_accuracy': train_metrics.get('mlm_accuracy', 0.0)
                }
                
                if val_metrics:
                    epoch_summary.update({
                        'epoch/val_loss': val_metrics.get('total_loss', 0.0),
                        'epoch/val_mlm_loss': val_metrics.get('mlm_loss', 0.0),
                        'epoch/val_alignment_loss': val_metrics.get('alignment_loss', 0.0),
                        'epoch/val_cls_similarity_positive': val_metrics.get('cls_cosine_sim_positive', 0.0),
                        'epoch/val_cls_similarity_negative': val_metrics.get('cls_cosine_sim_negative', 0.0),
                        'epoch/val_cls_similarity_gap': val_metrics.get('cls_cosine_sim_gap', 0.0),
                        'epoch/val_cls_similarity_ratio': val_metrics.get('cls_cosine_sim_ratio', 0.0),
                        'epoch/val_rankme_cls1': val_metrics.get('rankme_cls1', 0.0),
                        'epoch/val_rankme_cls2': val_metrics.get('rankme_cls2', 0.0),
                        'epoch/val_rankme_mean': val_metrics.get('rankme_mean', 0.0),
                        'epoch/val_rankme_mean_normalized': val_metrics.get('rankme_mean_normalized', 0.0)
                    })
                
                wandb.log(epoch_summary, step=self.global_step)
            
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
        
        # Log training completion to wandb
        if self.use_wandb and wandb.run is not None:
            wandb.summary['training_complete'] = True
            wandb.summary['total_epochs'] = num_epochs
            wandb.summary['total_steps'] = self.global_step
            
            # Create a summary table of all epochs
            if self.history:
                epoch_table = wandb.Table(
                    columns=['epoch', 'train_loss', 'train_mlm', 'train_align', 
                             'val_loss', 'val_mlm', 'val_align', 'cls_similarity']
                )
                for h in self.history:
                    epoch_table.add_data(
                        h['epoch'],
                        h['train_metrics'].get('total_loss', 0),
                        h['train_metrics'].get('mlm_loss', 0),
                        h['train_metrics'].get('alignment_loss', 0),
                        h['val_metrics'].get('total_loss', 0) if h['val_metrics'] else 0,
                        h['val_metrics'].get('mlm_loss', 0) if h['val_metrics'] else 0,
                        h['val_metrics'].get('alignment_loss', 0) if h['val_metrics'] else 0,
                        h['train_metrics'].get('cls_cosine_sim', 0)
                    )
                wandb.log({'training_summary': epoch_table})
        
        print("Training complete!")
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint to HuggingFace Hub if enabled, otherwise save locally.
        
        Args:
            filename: Checkpoint filename
        """
        # Push to HuggingFace Hub if enabled (skip local saving)
        if self.push_to_hub and self.hub_model_id:
            try:
                # Determine commit message with experiment metadata
                commit_parts = [f"Checkpoint: {filename} (epoch {self.epoch + 1}, step {self.global_step})"]
                if self.experiment_metadata.get('experiment_name'):
                    commit_parts.append(f"Experiment: {self.experiment_metadata['experiment_name']}")
                if self.experiment_metadata.get('config_description'):
                    commit_parts.append(self.experiment_metadata['config_description'])
                commit_message = " | ".join(commit_parts)
                
                # Push model to hub
                print(f"  → Pushing to HuggingFace Hub: {self.hub_model_id}")
                self.model.push_to_hub(
                    self.hub_model_id,
                    commit_message=commit_message,
                    private=False
                )
                
                # Push tokenizer to hub if available
                if self.tokenizer is not None:
                    self.tokenizer.push_to_hub(
                        self.hub_model_id,
                        commit_message=commit_message,
                        private=False
                    )
                
                hub_url = f"https://huggingface.co/{self.hub_model_id}"
                print(f"  ✓ Successfully pushed to Hub: {hub_url}")
                
                # Log to wandb with HuggingFace Hub reference
                if self.use_wandb and wandb.run is not None:
                    checkpoint_metadata = {
                        'checkpoint/name': filename,
                        'checkpoint/epoch': self.epoch + 1,
                        'checkpoint/global_step': self.global_step,
                        'checkpoint/best_val_loss': self.best_val_loss,
                        'checkpoint/storage': 'huggingface_hub',
                        'checkpoint/hub_url': hub_url,
                        'checkpoint/commit_message': commit_message
                    }
                    wandb.log(checkpoint_metadata, step=self.global_step)
                    
                    # Update wandb summary for best/final models
                    if filename == 'best_model.pt':
                        wandb.summary['best_model_hub_url'] = hub_url
                        wandb.summary['best_model_epoch'] = self.epoch + 1
                        wandb.summary['best_val_loss'] = self.best_val_loss
                    elif filename == 'final_model.pt':
                        wandb.summary['final_model_hub_url'] = hub_url
                        wandb.summary['final_model_epoch'] = self.epoch + 1
                
            except Exception as e:
                print(f"  ⚠ Warning: Failed to push to Hub: {e}")
                # Log the failure to wandb
                if self.use_wandb and wandb.run is not None:
                    wandb.log({
                        'checkpoint/error': str(e),
                        'checkpoint/name': filename,
                        'checkpoint/epoch': self.epoch + 1
                    }, step=self.global_step)
        else:
            # Save locally only if not pushing to hub
            checkpoint_path = os.path.join(self.save_dir, filename)
            
            # Ensure the save directory exists before saving
            try:
                os.makedirs(self.save_dir, exist_ok=True)
            except Exception as e:
                print(f"  ✗ Error creating directory {self.save_dir}: {e}")
                raise
            
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
            
            # Save checkpoint with error handling
            try:
                # Try to save to a temporary file first
                temp_checkpoint_path = checkpoint_path + '.tmp'
                torch.save(checkpoint, temp_checkpoint_path)
                # If successful, move to final location
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                os.rename(temp_checkpoint_path, checkpoint_path)
                print(f"  ✓ Saved checkpoint to {checkpoint_path}")
            except Exception as e:
                print(f"  ✗ Error saving checkpoint to {checkpoint_path}: {e}")
                print(f"  Directory exists: {os.path.exists(self.save_dir)}")
                print(f"  Directory writable: {os.access(self.save_dir, os.W_OK)}")
                # Try to get disk space information
                try:
                    disk_usage = shutil.disk_usage(self.save_dir)
                    print(f"  Disk space - Free: {disk_usage.free / (1024**3):.2f} GB, Total: {disk_usage.total / (1024**3):.2f} GB")
                except:
                    pass
                raise
            
            # Also save model in HuggingFace format
            model_dir = os.path.join(self.save_dir, filename.replace('.pt', ''))
            os.makedirs(model_dir, exist_ok=True)
            self.model.save_pretrained(model_dir)
            
            # Save tokenizer alongside model
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(model_dir)
            
            # Log checkpoint save to wandb
            if self.use_wandb and wandb.run is not None:
                checkpoint_metadata = {
                    'checkpoint/name': filename,
                    'checkpoint/epoch': self.epoch + 1,
                    'checkpoint/global_step': self.global_step,
                    'checkpoint/best_val_loss': self.best_val_loss,
                    'checkpoint/storage': 'local',
                    'checkpoint/path': checkpoint_path
                }
                wandb.log(checkpoint_metadata, step=self.global_step)
                
                # Save as wandb artifact (for best and final models)
                if filename in ['best_model.pt', 'final_model.pt']:
                    artifact_name = filename.replace('.pt', '').replace('_', '-')
                    artifact = wandb.Artifact(
                        name=f"{artifact_name}-{wandb.run.id}",
                        type='model',
                        description=f"Model checkpoint: {filename}",
                        metadata={
                            'epoch': self.epoch + 1,
                            'global_step': self.global_step,
                            'best_val_loss': self.best_val_loss,
                            'experiment_name': self.experiment_metadata.get('experiment_name', 'unknown'),
                            'config_description': self.experiment_metadata.get('config_description', '')
                        }
                    )
                    artifact.add_file(checkpoint_path)
                    artifact.add_dir(model_dir)
                    wandb.log_artifact(artifact)
                    print(f"  ✓ Logged artifact to wandb: {artifact_name}")
                
                # Update wandb summary for best/final models
                if filename == 'best_model.pt':
                    wandb.summary['best_model_path'] = checkpoint_path
                    wandb.summary['best_model_epoch'] = self.epoch + 1
                    wandb.summary['best_val_loss'] = self.best_val_loss
                elif filename == 'final_model.pt':
                    wandb.summary['final_model_path'] = checkpoint_path
                    wandb.summary['final_model_epoch'] = self.epoch + 1
    
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
