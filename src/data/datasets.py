"""
Dataset loader for WMT19 translation dataset with concatenated sequences.
Supports multiple language pairs for mixed batching.
"""
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
import random


def build_concatenated_sequence(
    text1: str,
    text2: str,
    tokenizer,
    max_length: int = 512
) -> Tuple[List[int], Dict[str, int]]:
    """
    Build concatenated sequence: [CLS] text1 [SEP] [CLS] text2 [SEP]
    
    Args:
        text1: First language text
        text2: Second language text
        tokenizer: XLM-RoBERTa tokenizer
        max_length: Maximum sequence length
    
    Returns:
        sequence_ids: List of token IDs
        positions: Dictionary tracking positions of special tokens and language ranges
    """
    # Tokenize both texts (without adding special tokens)
    tokens1 = tokenizer.tokenize(text1)
    tokens2 = tokenizer.tokenize(text2)
    
    # Truncate if needed (leave room for 4 special tokens: 2 CLS + 2 SEP)
    max_tokens_per_lang = (max_length - 4) // 2
    if len(tokens1) > max_tokens_per_lang:
        tokens1 = tokens1[:max_tokens_per_lang]
    if len(tokens2) > max_tokens_per_lang:
        tokens2 = tokens2[:max_tokens_per_lang]
    
    # Convert to IDs
    ids1 = tokenizer.convert_tokens_to_ids(tokens1)
    ids2 = tokenizer.convert_tokens_to_ids(tokens2)
    
    # Build sequence: [CLS] ids1 [SEP] [CLS] ids2 [SEP]
    sequence_ids = [
        tokenizer.cls_token_id,  # First CLS
        *ids1,                    # Language 1 tokens
        tokenizer.sep_token_id,   # First SEP
        tokenizer.cls_token_id,   # Second CLS
        *ids2,                    # Language 2 tokens
        tokenizer.sep_token_id    # Final SEP
    ]
    
    # Track positions
    first_cls_pos = 0
    lang1_start = 1
    lang1_end = 1 + len(ids1)
    first_sep_pos = lang1_end
    second_cls_pos = first_sep_pos + 1
    lang2_start = second_cls_pos + 1
    lang2_end = lang2_start + len(ids2)
    final_sep_pos = lang2_end
    
    positions = {
        'first_cls_pos': first_cls_pos,
        'lang1_start': lang1_start,
        'lang1_end': lang1_end,
        'first_sep_pos': first_sep_pos,
        'second_cls_pos': second_cls_pos,
        'lang2_start': lang2_start,
        'lang2_end': lang2_end,
        'final_sep_pos': final_sep_pos,
        'total_length': len(sequence_ids)
    }
    
    return sequence_ids, positions


class TranslationPairDataset(TorchDataset):
    """
    Dataset for translation pairs with concatenated sequences.
    Each example returns raw texts and language pair information.
    The collator handles tokenization and concatenation.
    """
    
    def __init__(
        self,
        examples: List[Dict[str, str]],
        lang_pair: str
    ):
        """
        Args:
            examples: List of dicts with 'text1' and 'text2' keys
            lang_pair: Language pair string (e.g., 'de-en')
        """
        self.examples = examples
        self.lang_pair = lang_pair
        self.lang1_code, self.lang2_code = lang_pair.split('-')
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'text1': example['text1'],
            'text2': example['text2'],
            'lang_pair': self.lang_pair,
            'lang1_code': self.lang1_code,
            'lang2_code': self.lang2_code
        }


class MultilingualDataset(TorchDataset):
    """
    Dataset combining multiple language pairs for mixed batching.
    """
    
    def __init__(self, datasets: List[TranslationPairDataset]):
        """
        Args:
            datasets: List of TranslationPairDataset objects
        """
        self.datasets = datasets
        
        # Create flat list of (dataset_idx, example_idx) tuples
        self.index_mapping = []
        for dataset_idx, dataset in enumerate(datasets):
            for example_idx in range(len(dataset)):
                self.index_mapping.append((dataset_idx, example_idx))
        
        # Shuffle to mix language pairs
        random.shuffle(self.index_mapping)
    
    def __len__(self):
        return len(self.index_mapping)
    
    def __getitem__(self, idx):
        dataset_idx, example_idx = self.index_mapping[idx]
        return self.datasets[dataset_idx][example_idx]


def load_translation_dataset(
    lang_pair: str,
    split: str = 'train',
    max_examples: Optional[int] = None,
    min_length: int = 10,
    max_length: int = 500
) -> TranslationPairDataset:
    """
    Load WMT19 translation dataset for a single language pair.
    
    Args:
        lang_pair: Language pair (e.g., 'de-en', 'cs-en')
        split: Dataset split ('train' or 'validation')
        max_examples: Maximum number of examples to load (None = all)
        min_length: Minimum text length (characters)
        max_length: Maximum text length (characters)
    
    Returns:
        TranslationPairDataset
    """
    # Load dataset from HuggingFace
    try:
        dataset = load_dataset("wmt19", lang_pair, split=split)
    except Exception as e:
        raise ValueError(f"Failed to load WMT19 dataset for {lang_pair}: {e}")
    
    lang1_code, lang2_code = lang_pair.split('-')
    
    # Extract and filter examples
    examples = []
    for i, example in enumerate(dataset):
        if max_examples and len(examples) >= max_examples:
            break
        
        translation = example.get('translation', {})
        if not isinstance(translation, dict):
            continue
        
        text1 = translation.get(lang1_code, '')
        text2 = translation.get(lang2_code, '')
        
        # Filter by length
        if (text1 and text2 and 
            len(text1) >= min_length and len(text2) >= min_length and
            len(text1) <= max_length and len(text2) <= max_length):
            
            examples.append({
                'text1': str(text1),
                'text2': str(text2)
            })
    
    if len(examples) == 0:
        raise ValueError(
            f"No valid examples found for {lang_pair}. "
            f"Check that the language codes are correct."
        )
    
    return TranslationPairDataset(examples, lang_pair)


def load_multilingual_dataset(
    lang_pairs: List[str],
    split: str = 'train',
    max_examples_per_pair: Optional[int] = None,
    min_length: int = 10,
    max_length: int = 500
) -> MultilingualDataset:
    """
    Load multiple language pairs into a single mixed dataset.
    
    Args:
        lang_pairs: List of language pairs (e.g., ['de-en', 'fr-en', 'cs-en'])
        split: Dataset split ('train' or 'validation')
        max_examples_per_pair: Maximum examples per language pair
        min_length: Minimum text length (characters)
        max_length: Maximum text length (characters)
    
    Returns:
        MultilingualDataset with mixed language pairs
    """
    datasets = []
    
    for lang_pair in lang_pairs:
        print(f"Loading {lang_pair}...")
        try:
            dataset = load_translation_dataset(
                lang_pair=lang_pair,
                split=split,
                max_examples=max_examples_per_pair,
                min_length=min_length,
                max_length=max_length
            )
            datasets.append(dataset)
            print(f"  Loaded {len(dataset)} examples")
        except Exception as e:
            print(f"  Warning: Failed to load {lang_pair}: {e}")
            continue
    
    if len(datasets) == 0:
        raise ValueError("No datasets loaded successfully")
    
    return MultilingualDataset(datasets)
