"""
Utility script to list all available language pairs in OPUS-100 dataset.
Helps you choose which language pairs to include in your training configuration.

Usage:
    python scripts/list_opus100_pairs.py [--sizes] [--filter LANG]
"""

import argparse
from datasets import get_dataset_config_names, load_dataset_builder


def get_split_sizes(config_name):
    """Get sizes for all splits without downloading the full dataset."""
    try:
        builder = load_dataset_builder("Helsinki-NLP/opus-100", config_name)
        info = builder.info
        
        sizes = {}
        if info.splits:
            for split_name, split_info in info.splits.items():
                sizes[split_name] = split_info.num_examples
        
        return sizes
    except Exception as e:
        return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="List available language pairs in OPUS-100 dataset"
    )
    parser.add_argument(
        '--sizes',
        action='store_true',
        help='Also fetch and display dataset sizes (uses metadata only, relatively fast)'
    )
    parser.add_argument(
        '--filter',
        type=str,
        help='Filter to show only pairs containing this language code (e.g., "en", "de")'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show train/validation/test split sizes separately (only with --sizes)'
    )
    args = parser.parse_args()
    
    print("Fetching available language pairs from OPUS-100...")
    print("=" * 100)
    
    try:
        # Get all available configurations (language pairs)
        configs = get_dataset_config_names("Helsinki-NLP/opus-100")
        
        # Apply filter if specified
        if args.filter:
            configs = [c for c in configs if args.filter in c.split('-')]
            print(f"Filtered to pairs containing '{args.filter}': {len(configs)} pairs found\n")
        else:
            print(f"Total language pairs available: {len(configs)}\n")
        
        # Sort for better readability
        configs = sorted(configs)
        
        if not args.sizes:
            # Just show the list organized by first language
            by_lang = {}
            for config in configs:
                lang1, lang2 = config.split('-')
                if lang1 not in by_lang:
                    by_lang[lang1] = []
                by_lang[lang1].append(lang2)
            
            print("Language pairs by source language:")
            print("-" * 100)
            
            for lang1 in sorted(by_lang.keys()):
                lang2_list = sorted(by_lang[lang1])
                print(f"{lang1:>3}: {', '.join(lang2_list)}")
            
            print("\n" + "=" * 100)
            print(f"Total pairs: {len(configs)}")
            print("\nTip: Use --sizes to see dataset sizes for each pair")
            print("     Use --filter en to show only English pairs")
        
        else:
            # Show sizes
            print("Fetching dataset sizes (using metadata, this will take a moment)...")
            print("-" * 100 + "\n")
            
            size_info = []
            for i, config in enumerate(configs):
                print(f"Progress: {i+1}/{len(configs)} - Fetching {config}...", end='\r')
                sizes = get_split_sizes(config)
                size_info.append((config, sizes))
            
            print(" " * 100, end='\r')  # Clear progress line
            
            # Calculate total examples per pair
            pairs_with_totals = []
            for config, sizes in size_info:
                if 'error' in sizes:
                    pairs_with_totals.append((config, 0, sizes))
                else:
                    total = sum(sizes.values())
                    pairs_with_totals.append((config, total, sizes))
            
            # Sort by total size (descending)
            pairs_with_totals.sort(key=lambda x: x[1], reverse=True)
            
            # Display results
            print("\n" + "=" * 100)
            print("OPUS-100 Dataset Sizes")
            print("=" * 100)
            
            if args.detailed:
                print(f"\n{'Language Pair':<15} {'Train':>15} {'Validation':>15} {'Test':>15} {'Total':>15}")
                print("-" * 100)
                for config, total, sizes in pairs_with_totals:
                    if 'error' in sizes:
                        print(f"{config:<15} {'Error':>15} {'':>15} {'':>15} {'':>15}")
                    else:
                        train = sizes.get('train', 0)
                        val = sizes.get('validation', 0)
                        test = sizes.get('test', 0)
                        print(f"{config:<15} {train:>15,} {val:>15,} {test:>15,} {total:>15,}")
            else:
                print(f"\n{'Language Pair':<15} {'Train Examples':>20} {'Total (all splits)':>25}")
                print("-" * 100)
                for config, total, sizes in pairs_with_totals:
                    if 'error' in sizes:
                        print(f"{config:<15} {'Error':>20} {'':>25}")
                    else:
                        train = sizes.get('train', 0)
                        print(f"{config:<15} {train:>20,} {total:>25,}")
            
            print("\n" + "=" * 100)
            print(f"Total language pairs: {len(configs)}")
            
            # Show summary statistics
            all_totals = [x[1] for x in pairs_with_totals if x[1] > 0]
            if all_totals:
                print(f"\nDataset Size Statistics:")
                print(f"  Largest:  {max(all_totals):,} examples")
                print(f"  Smallest: {min(all_totals):,} examples")
                print(f"  Average:  {sum(all_totals)//len(all_totals):,} examples")
                print(f"  Total:    {sum(all_totals):,} examples across all pairs")
            
            print("\n" + "=" * 100)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

