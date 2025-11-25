
import sys
import os
import torch
from transformers import XLMRobertaTokenizer

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.collators import DualObjectiveCollator
from src.models.bert_dual_objective import BertDualObjective

def test_collator_and_model():
    print("Initializing tokenizer...")
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    examples = [
        {'text1': 'Hello world', 'text2': 'Bonjour le monde', 'lang_pair': 'en-fr'},
        {'text1': 'Good morning', 'text2': 'Guten Morgen', 'lang_pair': 'en-de'}
    ]
    
    # Test Monolingual Strategy
    print("\n=== Testing Monolingual Strategy ===")
    collator_mono = DualObjectiveCollator(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        max_length=32,
        mlm_strategy='monolingual'
    )
    
    batch_mono = collator_mono(examples)
    print("Batch keys:", batch_mono.keys())
    
    # Verify keys
    expected_keys = [
        'lang1_input_ids', 'lang1_attention_mask', 
        'lang2_input_ids', 'lang2_attention_mask',
        'mlm_input_ids', 'mlm_attention_mask', 'mlm_labels',
        'mlm_strategy'
    ]
    for key in expected_keys:
        assert key in batch_mono, f"Missing key: {key}"
    
    print("✓ All expected keys present")
    print(f"lang1 shape: {batch_mono['lang1_input_ids'].shape}")
    print(f"mlm shape: {batch_mono['mlm_input_ids'].shape}")
    
    # Verify lang1 inputs are NOT masked (except padding)
    # 250001 is mask token for xlm-roberta-base
    mask_token_id = tokenizer.mask_token_id
    lang1_masks = (batch_mono['lang1_input_ids'] == mask_token_id).sum()
    print(f"Mask tokens in lang1_input_ids: {lang1_masks}")
    assert lang1_masks == 0, "lang1_input_ids should not contain mask tokens!"
    print("✓ lang1_input_ids are clean (unmasked)")
    
    # Verify MLM inputs ARE masked
    mlm_masks = (batch_mono['mlm_input_ids'] == mask_token_id).sum()
    print(f"Mask tokens in mlm_input_ids: {mlm_masks}")
    assert mlm_masks > 0, "mlm_input_ids should contain mask tokens!"
    print("✓ mlm_input_ids are masked")
    
    # Test Bilingual Strategy
    print("\n=== Testing Bilingual Strategy ===")
    collator_bi = DualObjectiveCollator(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        max_length=32,
        mlm_strategy='bilingual'
    )
    
    batch_bi = collator_bi(examples)
    print("Batch keys:", batch_bi.keys())
    print(f"mlm shape: {batch_bi['mlm_input_ids'].shape}")
    
    # Verify lang1 inputs are still clean
    lang1_masks = (batch_bi['lang1_input_ids'] == mask_token_id).sum()
    assert lang1_masks == 0, "lang1_input_ids should not contain mask tokens in bilingual mode!"
    print("✓ lang1_input_ids are clean (unmasked)")
    
    # Test Model Forward Pass
    print("\n=== Testing Model Forward Pass ===")
    # Create a small model for testing
    config = {
        'type': 'infonce',
        'temperature': 0.07
    }
    model = BertDualObjective(
        model_name='xlm-roberta-base',
        lambda_alignment=1.0,
        alignment_loss_config=config
    )
    
    # Move to CPU for test
    model.to('cpu')
    
    print("Running compute_total_loss with monolingual batch...")
    loss_mono, metrics_mono = model.compute_total_loss(batch_mono)
    print(f"Monolingual Loss: {loss_mono.item():.4f}")
    print("Metrics:", metrics_mono.keys())
    
    print("Running compute_total_loss with bilingual batch...")
    loss_bi, metrics_bi = model.compute_total_loss(batch_bi)
    print(f"Bilingual Loss: {loss_bi.item():.4f}")
    
    print("\n✓ Validation Successful!")

if __name__ == "__main__":
    test_collator_and_model()
