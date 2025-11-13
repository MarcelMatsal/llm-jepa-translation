"""
Quick test to verify a config file will load correctly.

Usage:
    python scripts/test_config.py <config_file>
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import argparse
from src.data.datasets import load_multilingual_dataset


def test_config(config_path):
    """Test that a config file loads and the dataset can be initialized."""
    
    print("=" * 80)
    print("Testing Configuration File")
    print("=" * 80)
    print(f"\nConfig file: {config_path}\n")
    
    # Load config
    print("Loading configuration...")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Config loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return 1
    
    # Display key settings
    print("-" * 80)
    print("Configuration Summary:")
    print("-" * 80)
    print(f"Experiment: {config.get('experiment_name', 'N/A')}")
    print(f"Base Model: {config['model']['base_model']}")
    print(f"Alignment Loss: {config['model']['alignment_loss_type']}")
    print(f"Lambda: {config['model']['lambda_alignment']}")
    print(f"\nLanguage Pairs: {config['data']['lang_pairs']}")
    print(f"Max Examples per Pair: {config['data']['max_examples_per_pair']}")
    print(f"Batch Size: {config['data']['batch_size']}")
    print(f"Max Length: {config['data']['max_length']}")
    print(f"\nEpochs: {config['training']['epochs']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print(f"Device: {config['output']['device']}")
    print("-" * 80)
    
    # Ask user if they want to test dataset loading
    print("\n" + "=" * 80)
    response = input("Test dataset loading with 100 examples per pair? (y/n): ").strip().lower()
    
    if response == 'y':
        print("=" * 80)
        print("\nTesting Dataset Loading (100 examples per pair)...")
        print("-" * 80 + "\n")
        
        try:
            dataset = load_multilingual_dataset(
                lang_pairs=config['data']['lang_pairs'],
                split='train',
                max_examples_per_pair=100,  # Just 100 for quick test
                min_length=config['data'].get('min_text_length', 10),
                max_length=config['data'].get('max_text_length', 500)
            )
            
            print("\n" + "=" * 80)
            print("✓ Dataset Loading Test PASSED")
            print("=" * 80)
            print(f"\nTotal examples loaded: {len(dataset)}")
            
            # Show sample
            if len(dataset) > 0:
                example = dataset[0]
                print(f"\nSample example:")
                print(f"  Pair: {example['lang_pair']}")
                print(f"  Text 1 ({example['lang1_code']}): {example['text1'][:80]}...")
                print(f"  Text 2 ({example['lang2_code']}): {example['text2'][:80]}...")
            
        except Exception as e:
            print("\n" + "=" * 80)
            print("❌ Dataset Loading Test FAILED")
            print("=" * 80)
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    print("\n" + "=" * 80)
    print("✓ All Tests Passed!")
    print("=" * 80)
    print("\nYou can now run training with:")
    print(f"  python scripts/train.py --config {config_path}")
    print("=" * 80)
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Test a configuration file")
    parser.add_argument('config', help='Path to config file')
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1
    
    return test_config(args.config)


if __name__ == "__main__":
    exit(main())

