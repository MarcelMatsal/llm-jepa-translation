"""
Masking strategies for dual-objective BERT training.
Implements standard MLM masking and cross-lingual masking for CLS token alignment.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple


def get_language_token_ranges(positions_dict: Dict) -> Tuple[range, range]:
    """
    Extract language token ranges from position dictionary.
    
    Args:
        positions_dict: Dictionary with keys:
            - lang1_start, lang1_end: first language token positions
            - lang2_start, lang2_end: second language token positions
    
    Returns:
        Tuple of (lang1_range, lang2_range)
    """
    lang1_range = range(positions_dict['lang1_start'], positions_dict['lang1_end'])
    lang2_range = range(positions_dict['lang2_start'], positions_dict['lang2_end'])
    return lang1_range, lang2_range


def create_mlm_mask(
    input_ids: torch.Tensor,
    positions_dict: Dict,
    tokenizer,
    mlm_probability: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create standard BERT-style MLM masking.
    Randomly masks ~15% of tokens (excluding special tokens like CLS, SEP).
    
    Args:
        input_ids: Input token IDs (batch_size, seq_len)
        positions_dict: Dictionary with language token positions
        tokenizer: Tokenizer with special token IDs
        mlm_probability: Probability of masking each token
    
    Returns:
        masked_input_ids: Input IDs with masked tokens
        labels: Labels for loss computation (-100 for non-masked positions)
    """
    masked_input_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)  # -100 = ignore in loss
    
    batch_size, seq_len = input_ids.shape
    
    for batch_idx in range(batch_size):
        # Get positions dict for this example
        if isinstance(positions_dict['lang1_start'], list):
            pos_dict = {k: v[batch_idx] for k, v in positions_dict.items()}
        else:
            pos_dict = positions_dict
        
        # Get special token positions
        special_positions = {
            pos_dict['first_cls_pos'],
            pos_dict['first_sep_pos'],
            pos_dict['second_cls_pos'],
            pos_dict['final_sep_pos']
        }
        
        # Get all token positions (excluding special tokens)
        lang1_range, lang2_range = get_language_token_ranges(pos_dict)
        maskable_positions = list(lang1_range) + list(lang2_range)
        
        # Randomly select tokens to mask
        mask_indices = np.random.binomial(1, mlm_probability, len(maskable_positions))
        
        for idx, should_mask in enumerate(mask_indices):
            if should_mask:
                pos = maskable_positions[idx]
                
                # Get original token
                original_token = input_ids[batch_idx, pos].item()
                
                # BERT masking strategy:
                # 80% of the time: replace with [MASK] token
                # 10% of the time: replace with random token
                # 10% of the time: keep original token
                rand = np.random.random()
                
                if rand < 0.8:
                    # Replace with [MASK]
                    masked_input_ids[batch_idx, pos] = tokenizer.mask_token_id
                elif rand < 0.9:
                    # Replace with random token
                    random_token = np.random.randint(0, tokenizer.vocab_size)
                    masked_input_ids[batch_idx, pos] = random_token
                # else: keep original token (10% of the time)
                
                # Set label for loss computation
                labels[batch_idx, pos] = original_token
    
    return masked_input_ids, labels


def create_cross_lingual_mask(
    input_ids: torch.Tensor,
    positions_dict: Dict,
    tokenizer,
    mask_language: int
) -> Tuple[torch.Tensor, List[int]]:
    """
    Create cross-lingual masking for CLS token extraction.
    
    Args:
        input_ids: Input token IDs (batch_size, seq_len)
        positions_dict: Dictionary with language token positions
        tokenizer: Tokenizer with special token IDs
        mask_language: Which language to mask (1 or 2)
    
    Returns:
        masked_input_ids: Input IDs with entire language masked
        cls_positions: List of CLS token positions to extract (one per batch item)
    """
    masked_input_ids = input_ids.clone()
    cls_positions = []
    
    batch_size = input_ids.shape[0]
    
    for batch_idx in range(batch_size):
        # Get positions dict for this example
        if isinstance(positions_dict['lang1_start'], list):
            pos_dict = {k: v[batch_idx] for k, v in positions_dict.items()}
        else:
            pos_dict = positions_dict
        
        if mask_language == 1:
            # Mask language 1 (keep language 2 visible)
            # Mask: [lang1_tokens + first_SEP]
            # Keep: [first_CLS + second_CLS + lang2_tokens + final_SEP]
            # Extract CLS from: second_cls_pos
            
            lang1_range, _ = get_language_token_ranges(pos_dict)
            
            # Mask all language 1 tokens
            for pos in lang1_range:
                masked_input_ids[batch_idx, pos] = tokenizer.mask_token_id
            
            # Mask first SEP
            masked_input_ids[batch_idx, pos_dict['first_sep_pos']] = tokenizer.mask_token_id
            
            # CLS to extract is the second CLS (before lang2)
            cls_positions.append(pos_dict['second_cls_pos'])
            
        elif mask_language == 2:
            # Mask language 2 (keep language 1 visible)
            # Mask: [second_CLS + lang2_tokens + final_SEP]
            # Keep: [first_CLS + lang1_tokens + first_SEP]
            # Extract CLS from: first_cls_pos (position 0)
            
            _, lang2_range = get_language_token_ranges(pos_dict)
            
            # Mask second CLS
            masked_input_ids[batch_idx, pos_dict['second_cls_pos']] = tokenizer.mask_token_id
            
            # Mask all language 2 tokens
            for pos in lang2_range:
                masked_input_ids[batch_idx, pos] = tokenizer.mask_token_id
            
            # Mask final SEP
            masked_input_ids[batch_idx, pos_dict['final_sep_pos']] = tokenizer.mask_token_id
            
            # CLS to extract is the first CLS (position 0)
            cls_positions.append(pos_dict['first_cls_pos'])
        else:
            raise ValueError(f"mask_language must be 1 or 2, got {mask_language}")
    
    return masked_input_ids, cls_positions

