"""
Data collator for dual-objective BERT training.
Supports configurable MLM strategies and clean monolingual encoding.
"""
import torch
from typing import Dict, List
from .datasets import build_concatenated_sequence
from .masking import create_mlm_mask, create_cross_lingual_mask


class DualObjectiveCollator:
    """
    Collator for dual-objective training with configurable MLM strategy.
    
    Supports two MLM strategies:
    - 'monolingual': Mask each language separately (matches inference, cleaner separation)
    - 'bilingual': Mask both languages together (XLM-TLM style, more alignment signal)
    
    Always prepares clean monolingual inputs for CLS extraction (fixing collapse issue).
    """
    
    def __init__(
        self,
        tokenizer,
        mlm_probability: float = 0.15,
        max_length: int = 512,
        pad_to_max_length: bool = False,
        mlm_strategy: str = 'monolingual'
    ):
        """
        Args:
            tokenizer: XLM-RoBERTa tokenizer
            mlm_probability: Probability of masking tokens for MLM
            max_length: Maximum sequence length
            pad_to_max_length: Whether to pad all sequences to max_length
            mlm_strategy: 'monolingual' or 'bilingual'
        """
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.mlm_strategy = mlm_strategy
        
        if mlm_strategy not in ['monolingual', 'bilingual']:
            raise ValueError(f"mlm_strategy must be 'monolingual' or 'bilingual', got {mlm_strategy}")
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.
        
        Args:
            examples: List of dicts with 'text1', 'text2', 'lang_pair' keys
        
        Returns:
            Dictionary containing:
                - lang1_input_ids, lang1_attention_mask: Clean lang1 input for CLS1
                - lang2_input_ids, lang2_attention_mask: Clean lang2 input for CLS2
                - mlm_input_ids, mlm_attention_mask, mlm_labels: For MLM loss
                - positions_dict: Position information (for bilingual MLM)
        """
        batch_size = len(examples)
        
        # === Prepare clean monolingual inputs for CLS extraction ===
        # This fixes the collapse issue by providing clean, unmasked inputs
        
        lang1_sequences = []
        lang2_sequences = []
        
        for example in examples:
            # Tokenize each language separately (clean monolingual inputs)
            tokens1 = self.tokenizer.tokenize(example['text1'])
            tokens2 = self.tokenizer.tokenize(example['text2'])
            
            # Truncate if needed (leave room for CLS + SEP)
            max_tokens = self.max_length - 2
            if len(tokens1) > max_tokens:
                tokens1 = tokens1[:max_tokens]
            if len(tokens2) > max_tokens:
                tokens2 = tokens2[:max_tokens]
            
            # Convert to IDs and add special tokens: [CLS] tokens [SEP]
            ids1 = [self.tokenizer.cls_token_id] + \
                   self.tokenizer.convert_tokens_to_ids(tokens1) + \
                   [self.tokenizer.sep_token_id]
            ids2 = [self.tokenizer.cls_token_id] + \
                   self.tokenizer.convert_tokens_to_ids(tokens2) + \
                   [self.tokenizer.sep_token_id]
            
            lang1_sequences.append(ids1)
            lang2_sequences.append(ids2)
        
        # Determine global max length for padding
        # We need both languages to have the same length to concatenate them for MLM
        max_len1 = max(len(seq) for seq in lang1_sequences)
        max_len2 = max(len(seq) for seq in lang2_sequences)
        max_len_global = max(max_len1, max_len2)
        
        if self.pad_to_max_length:
            max_len_global = self.max_length
        
        # Pad lang1 sequences
        lang1_input_ids = torch.full((batch_size, max_len_global), self.tokenizer.pad_token_id, dtype=torch.long)
        lang1_attention_mask = torch.zeros((batch_size, max_len_global), dtype=torch.long)
        
        for i, seq in enumerate(lang1_sequences):
            seq_len = len(seq)
            lang1_input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            lang1_attention_mask[i, :seq_len] = 1
        
        # Pad lang2 sequences
        lang2_input_ids = torch.full((batch_size, max_len_global), self.tokenizer.pad_token_id, dtype=torch.long)
        lang2_attention_mask = torch.zeros((batch_size, max_len_global), dtype=torch.long)
        
        for i, seq in enumerate(lang2_sequences):
            seq_len = len(seq)
            lang2_input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            lang2_attention_mask[i, :seq_len] = 1
        
        # === Prepare MLM inputs based on strategy ===
        
        if self.mlm_strategy == 'monolingual':
            # Monolingual MLM: Mask each language separately
            # This matches inference setup and provides cleaner separation
            
            # Create MLM masks for lang1
            mlm_input_ids_1, mlm_labels_1 = self._create_monolingual_mlm_mask(
                lang1_input_ids, lang1_attention_mask
            )
            
            # Create MLM masks for lang2
            mlm_input_ids_2, mlm_labels_2 = self._create_monolingual_mlm_mask(
                lang2_input_ids, lang2_attention_mask
            )
            
            # For simplicity in training loop, concatenate them
            # (trainer will process them separately or average losses)
            mlm_input_ids = torch.cat([mlm_input_ids_1, mlm_input_ids_2], dim=0)
            mlm_attention_mask = torch.cat([lang1_attention_mask, lang2_attention_mask], dim=0)
            mlm_labels = torch.cat([mlm_labels_1, mlm_labels_2], dim=0)
            
            batch_output = {
                # Clean monolingual inputs for CLS extraction
                'lang1_input_ids': lang1_input_ids,
                'lang1_attention_mask': lang1_attention_mask,
                'lang2_input_ids': lang2_input_ids,
                'lang2_attention_mask': lang2_attention_mask,
                
                # MLM inputs (both languages concatenated)
                'mlm_input_ids': mlm_input_ids,
                'mlm_attention_mask': mlm_attention_mask,
                'mlm_labels': mlm_labels,
                
                # Metadata
        'mlm_strategy': 'monolingual',
                'original_batch_size': batch_size,  # For splitting MLM losses
            }
            
        else:  # bilingual
            # Bilingual MLM: Mask both languages in concatenated sequence
            # This is like XLM-TLM, providing cross-lingual alignment signal
            
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
            if self.pad_to_max_length:
                max_seq_len = self.max_length
            
            input_ids = torch.full((batch_size, max_seq_len), self.tokenizer.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
            
            for i, seq in enumerate(all_sequences):
                seq_len = len(seq)
                input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
                attention_mask[i, :seq_len] = 1
            
            # Convert positions to batched format
            positions_dict = self._batch_positions(all_positions)
            
            # Create bilingual MLM mask
            mlm_input_ids, mlm_labels = create_mlm_mask(
                input_ids,
                positions_dict,
                self.tokenizer,
                self.mlm_probability
            )
            
            batch_output = {
                # Clean monolingual inputs for CLS extraction
                'lang1_input_ids': lang1_input_ids,
                'lang1_attention_mask': lang1_attention_mask,
                'lang2_input_ids': lang2_input_ids,
                'lang2_attention_mask': lang2_attention_mask,
                
                # Bilingual MLM inputs
                'mlm_input_ids': mlm_input_ids,
                'mlm_attention_mask': attention_mask,
                'mlm_labels': mlm_labels,
                
                # Position information (for reference)
                'positions_dict': positions_dict,
                
                # Metadata
                'mlm_strategy': 'bilingual',
                'original_batch_size': batch_size,
            }
        
        return batch_output
    
    def _create_monolingual_mlm_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create MLM mask for monolingual sequences.
        
        Args:
            input_ids: Input IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        
        Returns:
            masked_input_ids: Input with masks applied
            labels: Labels for MLM (-100 for non-masked)
        """
        batch_size, seq_len = input_ids.shape
        
        # Clone inputs
        masked_input_ids = input_ids.clone()
        labels = torch.full_like(input_ids, -100)
        
        for i in range(batch_size):
            # Get valid token positions (not padding, not special tokens)
            valid_positions = []
            for j in range(seq_len):
                if attention_mask[i, j] == 0:
                    continue  # Skip padding
                token_id = input_ids[i, j].item()
                if token_id in [self.tokenizer.cls_token_id, 
                               self.tokenizer.sep_token_id,
                               self.tokenizer.pad_token_id]:
                    continue  # Skip special tokens
                valid_positions.append(j)
            
            # Randomly select positions to mask
            num_to_mask = max(1, int(len(valid_positions) * self.mlm_probability))
            import random
            mask_positions = random.sample(valid_positions, min(num_to_mask, len(valid_positions)))
            
            # Apply masking strategy (80% mask, 10% random, 10% keep)
            for pos in mask_positions:
                labels[i, pos] = input_ids[i, pos]  # Save original token
                
                rand = random.random()
                if rand < 0.8:
                    # 80%: Replace with [MASK]
                    masked_input_ids[i, pos] = self.tokenizer.mask_token_id
                elif rand < 0.9:
                    # 10%: Replace with random token
                    masked_input_ids[i, pos] = random.randint(0, self.tokenizer.vocab_size - 1)
                # else: 10% keep original
        
        return masked_input_ids, labels
    
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
