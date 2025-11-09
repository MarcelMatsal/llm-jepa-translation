"""
Data collator for dual-objective BERT training.
Creates three versions of each batch: MLM masked, lang1 masked, lang2 masked.
"""
import torch
from typing import Dict, List
from .datasets import build_concatenated_sequence
from .masking import create_mlm_mask, create_cross_lingual_mask


class DualObjectiveCollator:
    """
    Collator that creates three versions of each batch:
    1. Standard MLM masked version (for MLM loss)
    2. Language 2 fully masked (for extracting CLS1 from lang1)
    3. Language 1 fully masked (for extracting CLS2 from lang2)
    """
    
    def __init__(
        self,
        tokenizer,
        mlm_probability: float = 0.15,
        max_length: int = 512,
        pad_to_max_length: bool = False
    ):
        """
        Args:
            tokenizer: XLM-RoBERTa tokenizer
            mlm_probability: Probability of masking tokens for MLM
            max_length: Maximum sequence length
            pad_to_max_length: Whether to pad all sequences to max_length
        """
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples into three versions.
        
        Args:
            examples: List of dicts with 'text1', 'text2', 'lang_pair' keys
        
        Returns:
            Dictionary containing:
                - mlm_input_ids, mlm_attention_mask, mlm_labels: For MLM loss
                - cls1_input_ids, cls1_attention_mask: For CLS1 extraction (lang2 masked)
                - cls2_input_ids, cls2_attention_mask: For CLS2 extraction (lang1 masked)
                - cls1_positions, cls2_positions: CLS token positions to extract
                - positions_dict: Full position information for each example
        """
        batch_size = len(examples)
        
        # Build concatenated sequences for all examples
        all_sequences = []
        all_positions = []
        
        for example in examples:
            sequence_ids, positions = build_concatenated_sequence(
                example['text1'],
                example['text2'],
                self.tokenizer,
                self.max_length
            )
            all_sequences.append(sequence_ids)
            all_positions.append(positions)
        
        # Pad sequences to same length within batch
        max_seq_len = max(len(seq) for seq in all_sequences)
        if self.pad_to_max_length:
            max_seq_len = self.max_length
        
        # Create padded tensors
        input_ids = torch.full(
            (batch_size, max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long
        )
        attention_mask = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.long
        )
        
        # Fill in sequences
        for i, seq in enumerate(all_sequences):
            seq_len = len(seq)
            input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
        
        # Convert positions to batched format
        positions_dict = self._batch_positions(all_positions)
        
        # === Version 1: Standard MLM masking ===
        mlm_input_ids, mlm_labels = create_mlm_mask(
            input_ids,
            positions_dict,
            self.tokenizer,
            self.mlm_probability
        )
        mlm_attention_mask = attention_mask.clone()
        
        # === Version 2: Language 2 masked (extract CLS1) ===
        cls1_input_ids, cls1_positions = create_cross_lingual_mask(
            input_ids,
            positions_dict,
            self.tokenizer,
            mask_language=2
        )
        cls1_attention_mask = attention_mask.clone()
        
        # === Version 3: Language 1 masked (extract CLS2) ===
        cls2_input_ids, cls2_positions = create_cross_lingual_mask(
            input_ids,
            positions_dict,
            self.tokenizer,
            mask_language=1
        )
        cls2_attention_mask = attention_mask.clone()
        
        return {
            # MLM version
            'mlm_input_ids': mlm_input_ids,
            'mlm_attention_mask': mlm_attention_mask,
            'mlm_labels': mlm_labels,
            
            # CLS1 version (lang2 masked, extract first CLS)
            'cls1_input_ids': cls1_input_ids,
            'cls1_attention_mask': cls1_attention_mask,
            'cls1_positions': torch.tensor(cls1_positions, dtype=torch.long),
            
            # CLS2 version (lang1 masked, extract second CLS)
            'cls2_input_ids': cls2_input_ids,
            'cls2_attention_mask': cls2_attention_mask,
            'cls2_positions': torch.tensor(cls2_positions, dtype=torch.long),
            
            # Position information
            'positions_dict': positions_dict,
            
            # Original unmasked version (for reference)
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
    
    def _batch_positions(self, positions_list: List[Dict[str, int]]) -> Dict[str, List[int]]:
        """
        Convert list of position dicts to batched format.
        
        Args:
            positions_list: List of position dictionaries
        
        Returns:
            Dictionary with lists of positions
        """
        batched = {}
        keys = positions_list[0].keys()
        
        for key in keys:
            batched[key] = [pos_dict[key] for pos_dict in positions_list]
        
        return batched


class SimpleCollator:
    """
    Simple collator for evaluation that only concatenates sequences.
    Does not create masked versions.
    """
    
    def __init__(self, tokenizer, max_length: int = 512):
        """
        Args:
            tokenizer: XLM-RoBERTa tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate examples without masking.
        
        Args:
            examples: List of dicts with 'text1', 'text2' keys
        
        Returns:
            Dictionary with input_ids, attention_mask, and positions
        """
        batch_size = len(examples)
        
        # Build concatenated sequences
        all_sequences = []
        all_positions = []
        
        for example in examples:
            sequence_ids, positions = build_concatenated_sequence(
                example['text1'],
                example['text2'],
                self.tokenizer,
                self.max_length
            )
            all_sequences.append(sequence_ids)
            all_positions.append(positions)
        
        # Pad sequences
        max_seq_len = max(len(seq) for seq in all_sequences)
        
        input_ids = torch.full(
            (batch_size, max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long
        )
        attention_mask = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.long
        )
        
        # Fill in sequences
        for i, seq in enumerate(all_sequences):
            seq_len = len(seq)
            input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
        
        # Convert positions to batched format
        positions_dict = {}
        keys = all_positions[0].keys()
        for key in keys:
            positions_dict[key] = [pos_dict[key] for pos_dict in all_positions]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'positions_dict': positions_dict
        }
