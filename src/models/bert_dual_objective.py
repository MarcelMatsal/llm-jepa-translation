"""
Dual-Objective BERT model for cross-lingual alignment.
Combines MLM loss with CLS token alignment loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaForMaskedLM, XLMRobertaModel
from typing import Dict, Optional, Tuple
import json
import os
from huggingface_hub import HfApi



class BertDualObjective(nn.Module):
    """
    BERT-style model with dual training objectives:
    1. Standard Masked Language Modeling (MLM)
    2. Cross-lingual CLS token alignment
    
    Architecture:
    - Base: XLM-RoBERTa
    - Loss: L_total = L_mlm + lambda * L_align
    """
    
    def __init__(
        self,
        model_name: str = 'xlm-roberta-base',
        lambda_alignment: float = 1.0,
        alignment_loss_type: str = 'mse',  # Deprecated, use alignment_loss_config
        alignment_loss_config: Optional[Dict] = None
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            lambda_alignment: Weight for alignment loss (default: 1.0)
            alignment_loss_type: DEPRECATED - Type of alignment loss ('mse', 'cosine', 'contrastive')
                Use alignment_loss_config instead
            alignment_loss_config: Dictionary with loss configuration
                Example: {'type': 'sigreg', 'num_slices': 1024}
        """
        super().__init__()
        
        # Load pre-trained model with MLM head
        self.mlm_model = XLMRobertaForMaskedLM.from_pretrained(model_name)
        
        # Also load base model for CLS extraction (shares weights with mlm_model)
        self.base_model = self.mlm_model.roberta
        
        # Configuration
        self.lambda_alignment = lambda_alignment
        self.d_model = self.base_model.config.hidden_size
        
        # Create alignment loss using factory
        from src.losses import create_loss
        
        if alignment_loss_config is None:
            # Backward compatibility: convert old format to new
            import warnings
            warnings.warn(
                f"Using deprecated 'alignment_loss_type' parameter. "
                f"Please use 'alignment_loss_config' instead. "
                f"Example: alignment_loss_config={{'type': '{alignment_loss_type}'}}",
                DeprecationWarning
            )
            alignment_loss_config = {
                'type': alignment_loss_type,
                'temperature': 0.07  # Default for contrastive losses
            }
        
        # Create loss and components
        self.alignment_loss_fn, alignment_components = create_loss(
            alignment_loss_config,
            embedding_dim=self.d_model
        )
        
        # Add components to model (if any)
        self.alignment_components = nn.ModuleDict(alignment_components)
        
        # Store update hooks for post-optimization updates
        self.update_hooks = self.alignment_loss_fn.get_update_hooks()
        
        # Store config for saving
        self.alignment_loss_config = alignment_loss_config
        self.alignment_loss_type = alignment_loss_config.get('type', alignment_loss_type)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Standard forward pass for MLM.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Labels for MLM loss (batch_size, seq_len), -100 for non-masked
        
        Returns:
            Dictionary with 'loss', 'logits', 'hidden_states'
        """
        outputs = self.mlm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        return {
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states[-1]  # Last layer
        }
    
    def extract_cls_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cls_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract CLS token embeddings from specified positions.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            cls_positions: CLS positions to extract (batch_size,)
        
        Returns:
            CLS embeddings (batch_size, d_model)
        """
        # Forward pass through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, d_model)
        
        # Extract CLS tokens from specified positions
        batch_size = hidden_states.shape[0]
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        cls_embeddings = hidden_states[batch_indices, cls_positions]  # (batch_size, d_model)
        
        return cls_embeddings
    
    def compute_mlm_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute standard MLM loss.
        
        Args:
            input_ids: Masked input token IDs
            attention_mask: Attention mask
            labels: True labels for masked positions
        
        Returns:
            loss: MLM loss
            metrics: Dictionary with metrics
        """
        outputs = self.forward(input_ids, attention_mask, labels)
        mlm_loss = outputs['loss']
        
        # Compute accuracy on masked tokens
        with torch.no_grad():
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
            
            # Only consider positions that were masked (labels != -100)
            mask = labels != -100
            if mask.sum() > 0:
                correct = (predictions[mask] == labels[mask]).float().sum()
                accuracy = correct / mask.sum()
            else:
                accuracy = torch.tensor(0.0)
        
        metrics = {
            'mlm_loss': mlm_loss.item() if mlm_loss is not None else 0.0,
            'mlm_accuracy': accuracy.item()
        }
        
        return mlm_loss, metrics
    
    def compute_alignment_loss(
        self,
        lang1_input_ids: torch.Tensor,
        lang1_attention_mask: torch.Tensor,
        lang2_input_ids: torch.Tensor,
        lang2_attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Compute CLS alignment loss between two language representations.
        
        FIX FOR COLLAPSE: Now uses clean, unmasked monolingual inputs instead of
        masked inputs. Each language is encoded separately, and CLS is extracted
        from position 0 (always the CLS token in monolingual encoding).
        
        Args:
            lang1_input_ids: Clean unmasked lang1 input: [CLS] lang1 [SEP]
            lang1_attention_mask: Attention mask for lang1
            lang2_input_ids: Clean unmasked lang2 input: [CLS] lang2 [SEP]
            lang2_attention_mask: Attention mask for lang2
        
        Returns:
            loss: Alignment loss
            cls1: CLS embeddings for language 1 (batch_size, d_model)
            cls2: CLS embeddings for language 2 (batch_size, d_model)
            metrics: Dictionary with metrics
        """
        batch_size = lang1_input_ids.shape[0]
        device = lang1_input_ids.device
        
        # Forward pass 1: Encode lang1 (clean, no masking)
        outputs1 = self.base_model(
            input_ids=lang1_input_ids,
            attention_mask=lang1_attention_mask
        )
        # Extract CLS from position 0
        cls1 = outputs1.last_hidden_state[:, 0, :]  # (batch_size, d_model)
        
        # Forward pass 2: Encode lang2 (clean, no masking)
        outputs2 = self.base_model(
            input_ids=lang2_input_ids,
            attention_mask=lang2_attention_mask
        )
        # Extract CLS from position 0
        cls2 = outputs2.last_hidden_state[:, 0, :]  # (batch_size, d_model)
        
        # Compute alignment loss using the modular loss function (e.g., InfoNCE)
        loss, metrics = self.alignment_loss_fn.compute(cls1, cls2)
        
        # Add standard metrics for monitoring
        with torch.no_grad():
            cls1_norm = F.normalize(cls1, p=2, dim=-1)
            cls2_norm = F.normalize(cls2, p=2, dim=-1)
            cosine_sim = F.cosine_similarity(cls1_norm, cls2_norm, dim=-1).mean()
            euclidean_dist = torch.norm(cls1_norm - cls2_norm, p=2, dim=-1).mean()
        
        # Update metrics with standard monitoring values
        metrics.update({
            'cls_cosine_sim': cosine_sim.item(),
            'cls_euclidean_dist': euclidean_dist.item()
        })
        
        return loss, cls1, cls2, metrics
    
    def compute_total_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss: L_total = L_mlm + lambda * L_align
        
        Args:
            batch: Dictionary from DualObjectiveCollator containing:
                - lang1_input_ids, lang1_attention_mask: Clean lang1 for CLS1
                - lang2_input_ids, lang2_attention_mask: Clean lang2 for CLS2
                - mlm_input_ids, mlm_attention_mask, mlm_labels: For MLM loss
                - mlm_strategy: 'monolingual' or 'bilingual'
                - original_batch_size: Batch size before MLM processing
        
        Returns:
            total_loss: Combined loss
            metrics: Dictionary with all metrics
        """
        # Compute MLM loss
        mlm_strategy = batch['mlm_strategy']
        
        if mlm_strategy == 'monolingual':
            # For monolingual MLM, batch contains both languages concatenated
            # We process them together and average the loss
            mlm_loss, mlm_metrics = self.compute_mlm_loss(
                batch['mlm_input_ids'],
                batch['mlm_attention_mask'],
                batch['mlm_labels']
            )
            # MLM loss is already averaged across all examples (both langs)
        else:  # bilingual
            # For bilingual MLM, process the concatenated sequence
            mlm_loss, mlm_metrics = self.compute_mlm_loss(
                batch['mlm_input_ids'],
                batch['mlm_attention_mask'],
                batch['mlm_labels']
            )
        
        # Compute alignment loss using clean monolingual inputs
        align_loss, cls1, cls2, align_metrics = self.compute_alignment_loss(
            batch['lang1_input_ids'],
            batch['lang1_attention_mask'],
            batch['lang2_input_ids'],
            batch['lang2_attention_mask']
        )
        
        # Combine losses
        total_loss = mlm_loss + self.lambda_alignment * align_loss
        
        # Combine metrics
        metrics = {
            'total_loss': total_loss.item(),
            'mlm_loss': mlm_metrics['mlm_loss'],
            'mlm_accuracy': mlm_metrics['mlm_accuracy'],
            'weighted_alignment_loss': (self.lambda_alignment * align_loss).item(),
            'cls_cosine_sim': align_metrics['cls_cosine_sim'],
            'cls_euclidean_dist': align_metrics['cls_euclidean_dist'],
            'mlm_strategy': mlm_strategy
        }
        
        # Add all alignment loss metrics (e.g., InfoNCE metrics)
        # This includes: alignment_loss, positive_sim, negative_sim, 
        # within_lang_neg_sim, cross_lang_neg_sim, contrastive_accuracy, etc.
        for key, value in align_metrics.items():
            if key not in metrics:  # Don't override existing keys
                metrics[key] = value
        
        return total_loss, metrics
    
    def get_cls_embeddings_for_eval(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positions_dict: Dict[str, list]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract both CLS tokens for evaluation (without masking).
        
        Args:
            input_ids: Unmasked input token IDs
            attention_mask: Attention mask
            positions_dict: Position dictionary with 'first_cls_pos' and 'second_cls_pos'
        
        Returns:
            cls1: First CLS embeddings (batch_size, d_model)
            cls2: Second CLS embeddings (batch_size, d_model)
        """
        # Forward pass
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, d_model)
        
        # Extract both CLS tokens
        batch_size = hidden_states.shape[0]
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        
        # First CLS (always at position 0)
        first_cls_positions = torch.tensor(
            positions_dict['first_cls_pos'],
            device=hidden_states.device
        )
        cls1 = hidden_states[batch_indices, first_cls_positions]
        
        # Second CLS
        second_cls_positions = torch.tensor(
            positions_dict['second_cls_pos'],
            device=hidden_states.device
        )
        cls2 = hidden_states[batch_indices, second_cls_positions]
        
        # Normalize
        cls1 = F.normalize(cls1, p=2, dim=-1)
        cls2 = F.normalize(cls2, p=2, dim=-1)
        
        return cls1, cls2
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        self.mlm_model.save_pretrained(save_directory)
        
        # Save additional config
        config = {
            'lambda_alignment': self.lambda_alignment,
            'alignment_loss_type': self.alignment_loss_type
        }
        with open(f"{save_directory}/dual_objective_config.json", 'w') as f:
            json.dump(config, f)
    
    def push_to_hub(
        self,
        repo_id: str,
        commit_message: Optional[str] = None,
        private: bool = False,
        **kwargs
    ):
        """
        Push model to HuggingFace Hub.
        
        Args:
            repo_id: Repository ID on HuggingFace Hub (e.g., 'username/model-name')
            commit_message: Commit message for the push
            private: Whether the repository should be private
            **kwargs: Additional arguments passed to push_to_hub
        """
        import tempfile
        import shutil
        
        # Create a temporary directory to save everything
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save model to temp directory
            self.save_pretrained(tmp_dir)
            
            # Push the base MLM model (includes config.json and pytorch_model.bin)
            self.mlm_model.push_to_hub(
                repo_id=repo_id,
                commit_message=commit_message or "Update model",
                private=private,
                **kwargs
            )
            
            # Upload the dual_objective_config.json separately
            api = HfApi()
            config_path = os.path.join(tmp_dir, "dual_objective_config.json")
            api.upload_file(
                path_or_fileobj=config_path,
                path_in_repo="dual_objective_config.json",
                repo_id=repo_id,
                commit_message=commit_message or "Update dual objective config",
                repo_type="model"
            )
    
    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load model from directory."""        
        # Load config
        with open(f"{load_directory}/dual_objective_config.json", 'r') as f:
            config = json.load(f)
        
        # Create model
        model = cls(
            model_name=load_directory,
            lambda_alignment=config['lambda_alignment'],
            alignment_loss_type=config['alignment_loss_type']
        )
        
        return model

