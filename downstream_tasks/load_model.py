"""
Simple script to load and test the trained dual-objective model.
Verifies that the architecture is compatible with standard BERT tasks.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import XLMRobertaTokenizer
from src.models.bert_dual_objective import BertDualObjective


def load_trained_model(checkpoint_path: str, device: str = 'cuda'):
    """
    Load the trained dual-objective model.
    
    Args:
        checkpoint_path: Path to saved model checkpoint
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded BertDualObjective model
        tokenizer: XLM-RoBERTa tokenizer
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Load model
    model = BertDualObjective.from_pretrained(checkpoint_path)
    
    # Load tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(checkpoint_path)
    
    # Move to device
    if device == 'cuda' and torch.cuda.is_available():
        model = model.to('cuda')
        print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        model = model.to('cpu')
        print("Model loaded on CPU")
    
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"  - Base architecture: XLM-RoBERTa")
    print(f"  - Hidden size: {model.d_model}")
    print(f"  - Lambda alignment: {model.lambda_alignment}")
    print(f"  - Alignment loss type: {model.alignment_loss_type}")
    
    return model, tokenizer


def verify_architecture(model, tokenizer):
    """
    Verify that the model has the correct BERT-compatible architecture.
    """
    print("\n" + "="*60)
    print("ARCHITECTURE VERIFICATION")
    print("="*60)
    
    # Check that we have access to base model
    print("\n1. Base Model Components:")
    print(f"   ✓ MLM Model: {type(model.mlm_model).__name__}")
    print(f"   ✓ Base Model: {type(model.base_model).__name__}")
    print(f"   ✓ Config: {model.base_model.config.model_type}")
    
    # Check model dimensions
    print("\n2. Model Dimensions:")
    print(f"   ✓ Hidden size: {model.d_model}")
    print(f"   ✓ Vocab size: {model.mlm_model.config.vocab_size}")
    print(f"   ✓ Max position embeddings: {model.mlm_model.config.max_position_embeddings}")
    print(f"   ✓ Number of layers: {model.mlm_model.config.num_hidden_layers}")
    print(f"   ✓ Number of attention heads: {model.mlm_model.config.num_attention_heads}")
    
    # Test tokenizer
    print("\n3. Tokenizer Test:")
    test_text = "This is a test sentence."
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.encode(test_text)
    print(f"   ✓ Tokenizer works: '{test_text}'")
    print(f"   ✓ Tokens: {tokens[:5]}... ({len(tokens)} total)")
    print(f"   ✓ Special tokens: CLS={tokenizer.cls_token_id}, SEP={tokenizer.sep_token_id}")
    
    # Test forward pass
    print("\n4. Forward Pass Test:")
    device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids]).to(device)
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    print(f"   ✓ Forward pass successful")
    print(f"   ✓ Output logits shape: {outputs['logits'].shape}")
    print(f"   ✓ Hidden states shape: {outputs['hidden_states'].shape}")
    
    # Test CLS extraction
    print("\n5. CLS Embedding Extraction Test:")
    with torch.no_grad():
        cls_embedding = model.extract_cls_embeddings(
            input_ids, 
            attention_mask, 
            torch.tensor([0]).to(device)
        )
    print(f"   ✓ CLS embedding shape: {cls_embedding.shape}")
    print(f"   ✓ CLS embedding norm: {torch.norm(cls_embedding).item():.4f}")
    
    print("\n" + "="*60)
    print("✅ ALL ARCHITECTURE CHECKS PASSED!")
    print("="*60)
    print("\nThe model is compatible with standard BERT/XLM-RoBERTa tasks.")
    print("You can use it for:")
    print("  - Sentence embeddings (CLS token)")
    print("  - Masked language modeling")
    print("  - Fine-tuning on downstream tasks")
    print("  - Cross-lingual similarity/retrieval")
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and verify trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to load model on"
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_trained_model(args.checkpoint, args.device)
    
    # Verify architecture
    verify_architecture(model, tokenizer)

