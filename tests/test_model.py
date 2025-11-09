"""
Tests for BertDualObjective model.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import XLMRobertaTokenizer

from src.models.bert_dual_objective import BertDualObjective
from src.data.collators import DualObjectiveCollator


def test_model_initialization():
    """Test model initialization."""
    print("\n" + "="*80)
    print("TEST: Model Initialization")
    print("="*80)
    
    model = BertDualObjective(
        model_name='xlm-roberta-base',
        lambda_alignment=1.0,
        alignment_loss_type='mse'
    )
    
    assert model.lambda_alignment == 1.0
    assert model.alignment_loss_type == 'mse'
    assert model.d_model == 768  # XLM-RoBERTa base hidden size
    
    print(f"✓ Model initialized")
    print(f"  Hidden size: {model.d_model}")
    print(f"  Lambda: {model.lambda_alignment}")
    print(f"  Loss type: {model.alignment_loss_type}")


def test_forward_pass():
    """Test forward pass for MLM."""
    print("\n" + "="*80)
    print("TEST: Forward Pass")
    print("="*80)
    
    model = BertDualObjective(model_name='xlm-roberta-base')
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    # Create dummy input
    text = "This is a test sentence"
    inputs = tokenizer(text, return_tensors='pt')
    
    # Forward pass
    outputs = model.forward(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask']
    )
    
    assert 'logits' in outputs
    assert 'hidden_states' in outputs
    
    # Check shapes
    batch_size, seq_len = inputs['input_ids'].shape
    vocab_size = tokenizer.vocab_size
    
    assert outputs['logits'].shape == (batch_size, seq_len, vocab_size)
    assert outputs['hidden_states'].shape == (batch_size, seq_len, model.d_model)
    
    print(f"✓ Forward pass successful")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Hidden states shape: {outputs['hidden_states'].shape}")


def test_cls_extraction():
    """Test CLS token extraction."""
    print("\n" + "="*80)
    print("TEST: CLS Extraction")
    print("="*80)
    
    model = BertDualObjective(model_name='xlm-roberta-base')
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    # Create batch with multiple sequences
    texts = ["Hello world", "Good morning"]
    inputs = tokenizer(texts, return_tensors='pt', padding=True)
    
    # Extract CLS from position 0
    cls_positions = torch.tensor([0, 0])  # Both at position 0
    cls_embeddings = model.extract_cls_embeddings(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        cls_positions=cls_positions
    )
    
    # Check shape
    batch_size = len(texts)
    assert cls_embeddings.shape == (batch_size, model.d_model)
    
    print(f"✓ CLS extraction successful")
    print(f"  CLS embeddings shape: {cls_embeddings.shape}")


def test_mlm_loss():
    """Test MLM loss computation."""
    print("\n" + "="*80)
    print("TEST: MLM Loss")
    print("="*80)
    
    model = BertDualObjective(model_name='xlm-roberta-base')
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    # Create dummy input with labels
    text = "This is a test"
    inputs = tokenizer(text, return_tensors='pt')
    
    # Create fake labels (mask first real token)
    labels = torch.full_like(inputs['input_ids'], -100)
    labels[0, 1] = inputs['input_ids'][0, 1]  # Label for position 1
    
    # Mask the token
    inputs['input_ids'][0, 1] = tokenizer.mask_token_id
    
    # Compute loss
    loss, metrics = model.compute_mlm_loss(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        labels=labels
    )
    
    assert loss is not None
    assert loss.item() > 0
    assert 'mlm_loss' in metrics
    assert 'mlm_accuracy' in metrics
    
    print(f"✓ MLM loss computed")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Accuracy: {metrics['mlm_accuracy']:.4f}")


def test_alignment_loss():
    """Test alignment loss computation."""
    print("\n" + "="*80)
    print("TEST: Alignment Loss")
    print("="*80)
    
    model = BertDualObjective(model_name='xlm-roberta-base')
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    # Create dummy inputs
    text1 = "Hello world"
    text2 = "Bonjour monde"
    
    inputs1 = tokenizer(text1, return_tensors='pt')
    inputs2 = tokenizer(text2, return_tensors='pt')
    
    cls_positions1 = torch.tensor([0])
    cls_positions2 = torch.tensor([0])
    
    # Compute alignment loss
    loss, cls1, cls2, metrics = model.compute_alignment_loss(
        cls1_input_ids=inputs1['input_ids'],
        cls1_attention_mask=inputs1['attention_mask'],
        cls1_positions=cls_positions1,
        cls2_input_ids=inputs2['input_ids'],
        cls2_attention_mask=inputs2['attention_mask'],
        cls2_positions=cls_positions2
    )
    
    assert loss is not None
    assert loss.item() >= 0
    assert cls1.shape == (1, model.d_model)
    assert cls2.shape == (1, model.d_model)
    assert 'alignment_loss' in metrics
    assert 'cls_cosine_sim' in metrics
    
    print(f"✓ Alignment loss computed")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Cosine similarity: {metrics['cls_cosine_sim']:.4f}")


def test_total_loss():
    """Test combined loss computation."""
    print("\n" + "="*80)
    print("TEST: Total Loss (MLM + Alignment)")
    print("="*80)
    
    model = BertDualObjective(model_name='xlm-roberta-base', lambda_alignment=1.0)
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    # Create dummy batch using collator
    examples = [
        {'text1': 'Hello world', 'text2': 'Bonjour le monde', 'lang_pair': 'en-fr'},
    ]
    
    collator = DualObjectiveCollator(tokenizer, mlm_probability=0.15)
    batch = collator(examples)
    
    # Compute total loss
    total_loss, metrics = model.compute_total_loss(batch)
    
    assert total_loss is not None
    assert total_loss.item() > 0
    assert 'total_loss' in metrics
    assert 'mlm_loss' in metrics
    assert 'alignment_loss' in metrics
    assert 'weighted_alignment_loss' in metrics
    assert 'cls_cosine_sim' in metrics
    
    # Check that total = mlm + lambda * alignment
    expected_total = metrics['mlm_loss'] + metrics['weighted_alignment_loss']
    assert abs(metrics['total_loss'] - expected_total) < 1e-5
    
    print(f"✓ Total loss computed")
    print(f"  Total loss: {metrics['total_loss']:.4f}")
    print(f"  MLM loss: {metrics['mlm_loss']:.4f}")
    print(f"  Alignment loss: {metrics['alignment_loss']:.4f}")
    print(f"  Weighted alignment: {metrics['weighted_alignment_loss']:.4f}")
    print(f"  CLS similarity: {metrics['cls_cosine_sim']:.4f}")


def test_gradient_flow():
    """Test that gradients flow correctly."""
    print("\n" + "="*80)
    print("TEST: Gradient Flow")
    print("="*80)
    
    model = BertDualObjective(model_name='xlm-roberta-base')
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    # Create batch
    examples = [
        {'text1': 'Test sentence one', 'text2': 'Phrase de test un', 'lang_pair': 'en-fr'},
    ]
    
    collator = DualObjectiveCollator(tokenizer, mlm_probability=0.15)
    batch = collator(examples)
    
    # Forward pass
    total_loss, metrics = model.compute_total_loss(batch)
    
    # Backward pass
    total_loss.backward()
    
    # Check that gradients exist
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    assert has_gradients, "No gradients found after backward pass"
    
    print(f"✓ Gradients flow correctly")
    print(f"  Loss: {total_loss.item():.4f}")


def test_eval_mode():
    """Test evaluation mode."""
    print("\n" + "="*80)
    print("TEST: Evaluation Mode")
    print("="*80)
    
    model = BertDualObjective(model_name='xlm-roberta-base')
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    model.eval()
    
    # Create dummy input
    from src.data.datasets import build_concatenated_sequence
    
    text1 = "Hello world"
    text2 = "Bonjour monde"
    
    sequence_ids, positions = build_concatenated_sequence(text1, text2, tokenizer)
    input_ids = torch.tensor([sequence_ids])
    attention_mask = torch.ones_like(input_ids)
    
    # Extract CLS embeddings
    with torch.no_grad():
        cls1, cls2 = model.get_cls_embeddings_for_eval(
            input_ids=input_ids,
            attention_mask=attention_mask,
            positions_dict={
                'first_cls_pos': [positions['first_cls_pos']],
                'second_cls_pos': [positions['second_cls_pos']]
            }
        )
    
    assert cls1.shape == (1, model.d_model)
    assert cls2.shape == (1, model.d_model)
    
    # Check normalization
    cls1_norm = torch.norm(cls1, p=2, dim=-1)
    cls2_norm = torch.norm(cls2, p=2, dim=-1)
    assert abs(cls1_norm.item() - 1.0) < 1e-5
    assert abs(cls2_norm.item() - 1.0) < 1e-5
    
    print(f"✓ Evaluation mode works")
    print(f"  CLS1 shape: {cls1.shape}, norm: {cls1_norm.item():.6f}")
    print(f"  CLS2 shape: {cls2.shape}, norm: {cls2_norm.item():.6f}")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("RUNNING MODEL TESTS")
    print("="*80)
    
    try:
        test_model_initialization()
        test_forward_pass()
        test_cls_extraction()
        test_mlm_loss()
        test_alignment_loss()
        test_total_loss()
        test_gradient_flow()
        test_eval_mode()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

