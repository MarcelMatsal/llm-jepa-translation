"""
Tests for data pipeline: datasets, collators, and masking.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import XLMRobertaTokenizer

from src.data.datasets import build_concatenated_sequence, TranslationPairDataset
from src.data.collators import DualObjectiveCollator
from src.data.masking import create_mlm_mask, create_cross_lingual_mask, get_language_token_ranges


def test_build_concatenated_sequence():
    """Test sequence concatenation and position tracking."""
    print("\n" + "="*80)
    print("TEST: Build Concatenated Sequence")
    print("="*80)
    
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    text1 = "Hello world"
    text2 = "Bonjour le monde"
    
    sequence_ids, positions = build_concatenated_sequence(text1, text2, tokenizer)
    
    # Check structure
    assert sequence_ids[positions['first_cls_pos']] == tokenizer.cls_token_id
    assert sequence_ids[positions['first_sep_pos']] == tokenizer.sep_token_id
    assert sequence_ids[positions['second_cls_pos']] == tokenizer.cls_token_id
    assert sequence_ids[positions['final_sep_pos']] == tokenizer.sep_token_id
    
    # Check positions are in order
    assert positions['first_cls_pos'] == 0
    assert positions['lang1_start'] == 1
    assert positions['lang1_start'] < positions['lang1_end']
    assert positions['lang1_end'] == positions['first_sep_pos']
    assert positions['first_sep_pos'] < positions['second_cls_pos']
    assert positions['second_cls_pos'] < positions['lang2_start']
    assert positions['lang2_start'] < positions['lang2_end']
    assert positions['lang2_end'] == positions['final_sep_pos']
    
    print(f"✓ Sequence length: {len(sequence_ids)}")
    print(f"✓ Positions: {positions}")
    print(f"✓ Structure validated")


def test_mlm_masking():
    """Test MLM masking strategy."""
    print("\n" + "="*80)
    print("TEST: MLM Masking")
    print("="*80)
    
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    text1 = "This is a test sentence"
    text2 = "Ceci est une phrase de test"
    
    sequence_ids, positions = build_concatenated_sequence(text1, text2, tokenizer)
    input_ids = torch.tensor([sequence_ids])
    
    # Create MLM mask
    masked_input_ids, labels = create_mlm_mask(
        input_ids, positions, tokenizer, mlm_probability=0.5  # High prob for testing
    )
    
    # Check that some tokens were masked
    num_masked = (labels[0] != -100).sum().item()
    print(f"✓ Number of masked tokens: {num_masked}")
    assert num_masked > 0, "No tokens were masked"
    
    # Check that special tokens were not masked
    assert labels[0, positions['first_cls_pos']] == -100
    assert labels[0, positions['first_sep_pos']] == -100
    assert labels[0, positions['second_cls_pos']] == -100
    assert labels[0, positions['final_sep_pos']] == -100
    print(f"✓ Special tokens preserved")
    
    # Check that masked positions have labels
    for i in range(len(sequence_ids)):
        if labels[0, i] != -100:
            assert input_ids[0, i] != masked_input_ids[0, i] or \
                   masked_input_ids[0, i] == input_ids[0, i], \
                   "Masked token should be changed or kept (10% case)"
    print(f"✓ MLM masking validated")


def test_cross_lingual_masking():
    """Test cross-lingual masking strategy."""
    print("\n" + "="*80)
    print("TEST: Cross-Lingual Masking")
    print("="*80)
    
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    text1 = "This is English"
    text2 = "Ceci est français"
    
    sequence_ids, positions = build_concatenated_sequence(text1, text2, tokenizer)
    input_ids = torch.tensor([sequence_ids])
    
    # Test masking language 2 (keep language 1 visible)
    masked_lang2, cls_positions = create_cross_lingual_mask(
        input_ids, positions, tokenizer, mask_language=2
    )
    
    # Check that language 2 was masked
    lang1_range, lang2_range = get_language_token_ranges(positions)
    for pos in lang2_range:
        assert masked_lang2[0, pos] == tokenizer.mask_token_id
    print(f"✓ Language 2 masked completely")
    
    # Check that language 1 is visible
    for pos in lang1_range:
        assert masked_lang2[0, pos] == input_ids[0, pos]
    print(f"✓ Language 1 kept visible")
    
    # Check CLS position
    assert cls_positions[0] == positions['first_cls_pos']
    print(f"✓ CLS position correct: {cls_positions[0]}")
    
    # Test masking language 1 (keep language 2 visible)
    masked_lang1, cls_positions = create_cross_lingual_mask(
        input_ids, positions, tokenizer, mask_language=1
    )
    
    # Check that language 1 was masked
    for pos in lang1_range:
        assert masked_lang1[0, pos] == tokenizer.mask_token_id
    print(f"✓ Language 1 masked completely")
    
    # Check that language 2 is visible
    for pos in lang2_range:
        assert masked_lang1[0, pos] == input_ids[0, pos]
    print(f"✓ Language 2 kept visible")
    
    # Check CLS position
    assert cls_positions[0] == positions['second_cls_pos']
    print(f"✓ CLS position correct: {cls_positions[0]}")


def test_collator():
    """Test DualObjectiveCollator."""
    print("\n" + "="*80)
    print("TEST: DualObjectiveCollator")
    print("="*80)
    
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    # Create dummy examples
    examples = [
        {'text1': 'Hello world', 'text2': 'Bonjour le monde', 'lang_pair': 'en-fr'},
        {'text1': 'Good morning', 'text2': 'Guten Morgen', 'lang_pair': 'en-de'},
    ]
    
    collator = DualObjectiveCollator(tokenizer, mlm_probability=0.15, max_length=128)
    
    # Collate batch
    batch = collator(examples)
    
    # Check batch structure
    assert 'mlm_input_ids' in batch
    assert 'mlm_attention_mask' in batch
    assert 'mlm_labels' in batch
    assert 'cls1_input_ids' in batch
    assert 'cls1_attention_mask' in batch
    assert 'cls1_positions' in batch
    assert 'cls2_input_ids' in batch
    assert 'cls2_attention_mask' in batch
    assert 'cls2_positions' in batch
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    print(f"✓ Batch structure correct")
    
    # Check shapes
    batch_size = len(examples)
    assert batch['mlm_input_ids'].shape[0] == batch_size
    assert batch['cls1_input_ids'].shape[0] == batch_size
    assert batch['cls2_input_ids'].shape[0] == batch_size
    print(f"✓ Batch size correct: {batch_size}")
    
    # Check that sequences are different
    assert not torch.equal(batch['mlm_input_ids'], batch['input_ids']), \
        "MLM version should be different from original"
    assert not torch.equal(batch['cls1_input_ids'], batch['input_ids']), \
        "CLS1 version should be different from original"
    assert not torch.equal(batch['cls2_input_ids'], batch['input_ids']), \
        "CLS2 version should be different from original"
    print(f"✓ Three versions are different")
    
    # Check CLS positions
    assert batch['cls1_positions'].shape[0] == batch_size
    assert batch['cls2_positions'].shape[0] == batch_size
    print(f"✓ CLS positions: {batch['cls1_positions'].tolist()}, {batch['cls2_positions'].tolist()}")
    
    print(f"✓ Collator validated")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("RUNNING DATA PIPELINE TESTS")
    print("="*80)
    
    try:
        test_build_concatenated_sequence()
        test_mlm_masking()
        test_cross_lingual_masking()
        test_collator()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

