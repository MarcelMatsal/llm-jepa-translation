"""
Quick test script to verify OPUS-100 dataset loading works correctly.

Usage:
    python scripts/test_opus100_loading.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.datasets import load_multilingual_dataset


def test_opus100_loading():
    """Test loading a few language pairs from OPUS-100."""
    
    print("=" * 80)
    print("Testing OPUS-100 Dataset Loading")
    print("=" * 80)
    
    # Test with a small number of examples from diverse language pairs
    test_lang_pairs = [
        "de-en",  # German-English (large dataset)
        "fr-en",  # French-English (large dataset)
        "en-ja",  # English-Japanese (large dataset)
    ]
    
    print(f"\nAttempting to load language pairs: {test_lang_pairs}")
    print(f"Max examples per pair: 100 (for quick testing)")
    print("\n" + "-" * 80 + "\n")
    
    try:
        # Load dataset with small limit for testing
        dataset = load_multilingual_dataset(
            lang_pairs=test_lang_pairs,
            split='train',
            max_examples_per_pair=100,  # Just 100 examples per pair for quick test
            min_length=10,
            max_length=500
        )
        
        print("\n" + "=" * 80)
        print("✓ SUCCESS: Dataset loaded successfully!")
        print("=" * 80)
        print(f"\nTotal examples in combined dataset: {len(dataset)}")
        
        # Sample a few examples
        print("\n" + "-" * 80)
        print("Sample examples:")
        print("-" * 80)
        
        for i in range(min(3, len(dataset))):
            example = dataset[i]
            print(f"\nExample {i+1}:")
            print(f"  Language pair: {example['lang_pair']}")
            print(f"  Text 1 ({example['lang1_code']}): {example['text1'][:80]}...")
            print(f"  Text 2 ({example['lang2_code']}): {example['text2'][:80]}...")
        
        print("\n" + "=" * 80)
        print("All tests passed! ✓")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ FAILED: Error loading dataset")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(test_opus100_loading())

