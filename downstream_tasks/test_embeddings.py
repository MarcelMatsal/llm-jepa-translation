"""
Test extracting sentence embeddings from the trained model.
This demonstrates using the model like a standard BERT/sentence-transformer.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from transformers import XLMRobertaTokenizer
from src.models.bert_dual_objective import BertDualObjective
from scipy.spatial.distance import cosine


def get_sentence_embedding(text: str, model, tokenizer, device='cuda'):
    """
    Extract CLS embedding for a single sentence.
    
    Args:
        text: Input text
        model: Trained model
        tokenizer: Tokenizer
        device: Device
    
    Returns:
        embedding: Normalized CLS embedding (d_model,)
    """
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
    
    # Extract CLS embedding (position 0)
    with torch.no_grad():
        cls_positions = torch.tensor([0]).to(device)
        embedding = model.extract_cls_embeddings(input_ids, attention_mask, cls_positions)
    
    return embedding.cpu().numpy()[0]


def test_sentence_similarity(model, tokenizer, device='cuda'):
    """
    Test sentence similarity using CLS embeddings.
    """
    print("\n" + "="*60)
    print("SENTENCE SIMILARITY TEST")
    print("="*60)
    
    # Test sentences
    sentences = {
        'en1': "The cat sits on the mat.",
        'en2': "A cat is sitting on a mat.",
        'en3': "The weather is nice today.",
        'de1': "Die Katze sitzt auf der Matte.",  # Translation of en1
        'fr1': "Le chat est assis sur le tapis.",  # Translation of en1
    }
    
    # Get embeddings
    print("\nExtracting embeddings...")
    embeddings = {}
    for key, text in sentences.items():
        emb = get_sentence_embedding(text, model, tokenizer, device)
        embeddings[key] = emb
        print(f"  ✓ {key}: {text[:50]}")
    
    # Compute similarities
    print("\nSimilarity Matrix (Cosine Similarity):")
    print("-" * 60)
    
    keys = list(sentences.keys())
    print(f"{'':>10}", end="")
    for k in keys:
        print(f"{k:>10}", end="")
    print()
    
    for k1 in keys:
        print(f"{k1:>10}", end="")
        for k2 in keys:
            sim = 1 - cosine(embeddings[k1], embeddings[k2])
            print(f"{sim:>10.4f}", end="")
        print()
    
    # Key observations
    print("\n" + "="*60)
    print("KEY OBSERVATIONS:")
    print("="*60)
    
    # Similar English sentences
    sim_en_similar = 1 - cosine(embeddings['en1'], embeddings['en2'])
    print(f"\n1. Similar English sentences (en1 ↔ en2):")
    print(f"   Similarity: {sim_en_similar:.4f}")
    
    # Different English sentences
    sim_en_different = 1 - cosine(embeddings['en1'], embeddings['en3'])
    print(f"\n2. Different English sentences (en1 ↔ en3):")
    print(f"   Similarity: {sim_en_different:.4f}")
    
    # Cross-lingual (English-German)
    sim_en_de = 1 - cosine(embeddings['en1'], embeddings['de1'])
    print(f"\n3. Cross-lingual translation (en1 ↔ de1):")
    print(f"   Similarity: {sim_en_de:.4f}")
    print(f"   Expected: High (these are translations)")
    
    # Cross-lingual (English-French)
    sim_en_fr = 1 - cosine(embeddings['en1'], embeddings['fr1'])
    print(f"\n4. Cross-lingual translation (en1 ↔ fr1):")
    print(f"   Similarity: {sim_en_fr:.4f}")
    print(f"   Expected: High (these are translations)")
    
    # Random cross-lingual
    sim_random = 1 - cosine(embeddings['en3'], embeddings['de1'])
    print(f"\n5. Non-translation pair (en3 ↔ de1):")
    print(f"   Similarity: {sim_random:.4f}")
    print(f"   Expected: Lower than translations")
    
    print("\n" + "="*60)
    print("✅ EMBEDDING EXTRACTION SUCCESSFUL!")
    print("="*60)


def test_custom_sentences(model, tokenizer, device='cuda'):
    """
    Interactive test with custom sentences.
    """
    print("\n" + "="*60)
    print("CUSTOM SENTENCE TEST")
    print("="*60)
    print("\nEnter two sentences to compare (or press Enter to skip):\n")
    
    text1 = input("Sentence 1: ").strip()
    if not text1:
        print("Skipping custom test.")
        return
    
    text2 = input("Sentence 2: ").strip()
    if not text2:
        print("Skipping custom test.")
        return
    
    # Get embeddings
    emb1 = get_sentence_embedding(text1, model, tokenizer, device)
    emb2 = get_sentence_embedding(text2, model, tokenizer, device)
    
    # Compute similarity
    similarity = 1 - cosine(emb1, emb2)
    
    print(f"\nResults:")
    print(f"  Sentence 1: {text1}")
    print(f"  Sentence 2: {text2}")
    print(f"  Cosine Similarity: {similarity:.4f}")
    print(f"  Euclidean Distance: {np.linalg.norm(emb1 - emb2):.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test sentence embeddings")
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
        help="Run interactive custom sentence test"
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
    test_sentence_similarity(model, tokenizer, args.device)
    
    if args.custom:
        test_custom_sentences(model, tokenizer, args.device)

