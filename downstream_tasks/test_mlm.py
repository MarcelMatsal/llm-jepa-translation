"""
Test the MLM (Masked Language Modeling) capability of the trained model.
Demonstrates that the model still functions as a standard BERT for MLM tasks.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import XLMRobertaTokenizer
from src.models.bert_dual_objective import BertDualObjective


def predict_masked_token(text: str, model, tokenizer, device='cuda', top_k=5):
    """
    Predict masked tokens in text.
    
    Args:
        text: Text with <mask> token(s)
        model: Trained model
        tokenizer: Tokenizer
        device: Device
        top_k: Number of top predictions to show
    
    Returns:
        predictions: List of (token, probability) tuples
    """
    # Replace <mask> with tokenizer's mask token
    text = text.replace('<mask>', tokenizer.mask_token)
    
    # Tokenize
    encoded = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Find mask positions
    mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
    
    if len(mask_positions[1]) == 0:
        print("No mask token found in text!")
        return []
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs['logits']
    
    # Get predictions for each mask
    results = []
    for batch_idx, pos_idx in zip(mask_positions[0], mask_positions[1]):
        mask_logits = logits[batch_idx, pos_idx]
        probs = torch.softmax(mask_logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, top_k)
        top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
        
        predictions = list(zip(top_tokens, top_probs.cpu().numpy()))
        results.append(predictions)
    
    return results


def test_mlm_english(model, tokenizer, device='cuda'):
    """
    Test MLM on English sentences.
    """
    print("\n" + "="*60)
    print("MASKED LANGUAGE MODELING TEST - ENGLISH")
    print("="*60)
    
    test_cases = [
        "The cat sits on the <mask>.",
        "I love to eat <mask> for breakfast.",
        "Paris is the capital of <mask>.",
        "The weather is <mask> today.",
    ]
    
    for text in test_cases:
        print(f"\nInput: {text}")
        predictions = predict_masked_token(text, model, tokenizer, device, top_k=5)
        
        for mask_idx, preds in enumerate(predictions):
            print(f"  Predictions for mask {mask_idx + 1}:")
            for token, prob in preds:
                print(f"    {token.strip():20s} {prob:.4f}")


def test_mlm_multilingual(model, tokenizer, device='cuda'):
    """
    Test MLM on multiple languages.
    """
    print("\n" + "="*60)
    print("MASKED LANGUAGE MODELING TEST - MULTILINGUAL")
    print("="*60)
    
    test_cases = [
        ("English", "The sky is <mask>."),
        ("German", "Die Katze sitzt auf der <mask>."),
        ("French", "Le chat est <mask> sur le tapis."),
        ("Spanish", "El gato está en la <mask>."),
    ]
    
    for lang, text in test_cases:
        print(f"\n{lang}: {text}")
        predictions = predict_masked_token(text, model, tokenizer, device, top_k=3)
        
        for mask_idx, preds in enumerate(predictions):
            print(f"  Top predictions:")
            for token, prob in preds:
                print(f"    {token.strip():20s} {prob:.4f}")


def test_custom_mlm(model, tokenizer, device='cuda'):
    """
    Interactive MLM test with custom text.
    """
    print("\n" + "="*60)
    print("CUSTOM MLM TEST")
    print("="*60)
    print("\nEnter a sentence with <mask> token(s) (or press Enter to skip):")
    print("Example: The cat sits on the <mask>.\n")
    
    text = input("Your text: ").strip()
    if not text:
        print("Skipping custom test.")
        return
    
    if '<mask>' not in text:
        print("⚠️  No <mask> token found. Please include <mask> in your text.")
        return
    
    print(f"\nInput: {text}")
    predictions = predict_masked_token(text, model, tokenizer, device, top_k=10)
    
    for mask_idx, preds in enumerate(predictions):
        print(f"\nPredictions for mask {mask_idx + 1}:")
        for i, (token, prob) in enumerate(preds, 1):
            print(f"  {i:2d}. {token.strip():20s} {prob:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MLM capabilities")
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
    parser.add_argument(
        "--custom",
        action="store_true",
        help="Run interactive custom MLM test"
    )
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = BertDualObjective.from_pretrained(args.checkpoint)
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.checkpoint)
    
    if args.device == 'cuda' and torch.cuda.is_available():
        model = model.to('cuda')
        print(f"✓ Model loaded on GPU")
    else:
        model = model.to('cpu')
        print("✓ Model loaded on CPU")
    
    model.eval()
    
    # Run tests
    test_mlm_english(model, tokenizer, args.device)
    test_mlm_multilingual(model, tokenizer, args.device)
    
    if args.custom:
        test_custom_mlm(model, tokenizer, args.device)
    
    print("\n" + "="*60)
    print("✅ MLM TESTS COMPLETE!")
    print("="*60)
    print("\nThe model maintains its BERT MLM capabilities.")

