"""
Flexible dataset loaders for various HuggingFace translation datasets.
"""
from typing import Dict, List, Optional, Tuple
from datasets import Dataset, load_dataset
from torch.utils.data import Dataset as TorchDataset


class TranslationDataset(TorchDataset):
    """
    Dataset for translation pairs with language IDs.
    
    Supports multiple formats:
    - HuggingFace translation datasets (e.g., wmt19, opus_books)
    - Custom JSONL with 'text_src', 'text_tgt', 'lang_src', 'lang_tgt'
    - Any dataset with processor function
    """
    
    def __init__(
        self,
        texts_src: List[str],
        texts_tgt: List[str],
        lang_src: List[int],
        lang_tgt: List[int]
    ):
        """
        Args:
            texts_src: Source language texts
            texts_tgt: Target language texts
            lang_src: Source language IDs (mapped to integers)
            lang_tgt: Target language IDs (mapped to integers)
        """
        assert len(texts_src) == len(texts_tgt) == len(lang_src) == len(lang_tgt)
        self.texts_src = texts_src
        self.texts_tgt = texts_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
    
    def __len__(self):
        return len(self.texts_src)
    
    def __getitem__(self, idx):
        return {
            'text_src': self.texts_src[idx],
            'text_tgt': self.texts_tgt[idx],
            'lang_src': self.lang_src[idx],
            'lang_tgt': self.lang_tgt[idx]
        }


def process_translation_dataset(
    dataset: Dataset,
    lang_pair: str,
    lang_map: Dict[str, int],
    split: str = 'train'
) -> TranslationDataset:
    """
    Process HuggingFace translation dataset (e.g., wmt19, opus_books).
    
    Args:
        dataset: HuggingFace Dataset object
        lang_pair: Language pair like 'en-de' (source-target)
        lang_map: Mapping from language codes to integer IDs
        split: Dataset split ('train', 'validation', 'test')
    
    Returns:
        TranslationDataset
    """
    if split in dataset:
        split_dataset = dataset[split]
    else:
        split_dataset = dataset['train']
    
    lang_src_code, lang_tgt_code = lang_pair.split('-')
    
    # Extract texts and language IDs
    texts_src = []
    texts_tgt = []
    lang_src_ids = []
    lang_tgt_ids = []
    
    for example in split_dataset:
        text_src = example.get(lang_src_code, '')
        text_tgt = example.get(lang_tgt_code, '')
        
        if text_src and text_tgt:  # Skip empty pairs
            texts_src.append(text_src)
            texts_tgt.append(text_tgt)
            lang_src_ids.append(lang_map[lang_src_code])
            lang_tgt_ids.append(lang_map[lang_tgt_code])
    
    return TranslationDataset(texts_src, texts_tgt, lang_src_ids, lang_tgt_ids)


def process_custom_jsonl(
    file_path: str,
    lang_map: Dict[str, int],
    text_src_key: str = 'text_src',
    text_tgt_key: str = 'text_tgt',
    lang_src_key: str = 'lang_src',
    lang_tgt_key: str = 'lang_tgt'
) -> TranslationDataset:
    """
    Process custom JSONL file.
    
    Args:
        file_path: Path to JSONL file
        lang_map: Mapping from language codes to integer IDs
        text_src_key: Key for source text in JSON
        text_tgt_key: Key for target text in JSON
        lang_src_key: Key for source language in JSON
        lang_tgt_key: Key for target language in JSON
    
    Returns:
        TranslationDataset
    """
    import json
    
    texts_src = []
    texts_tgt = []
    lang_src_ids = []
    lang_tgt_ids = []
    
    with open(file_path, 'r') as f:
        for line in f:
            example = json.loads(line)
            texts_src.append(example[text_src_key])
            texts_tgt.append(example[text_tgt_key])
            lang_src_ids.append(lang_map[example[lang_src_key]])
            lang_tgt_ids.append(lang_map[example[lang_tgt_key]])
    
    return TranslationDataset(texts_src, texts_tgt, lang_src_ids, lang_tgt_ids)


def get_dataset(
    dataset_name: str,
    lang_pair: Optional[str] = None,
    lang_map: Optional[Dict[str, int]] = None,
    split: str = 'train',
    processor_fn: Optional[callable] = None,
    **kwargs
) -> TranslationDataset:
    """
    Unified dataset loader supporting multiple formats.
    
    Args:
        dataset_name: HuggingFace dataset name or path to JSONL file
        lang_pair: Language pair (e.g., 'en-de') for HF datasets
        lang_map: Language code to integer ID mapping
        split: Dataset split
        processor_fn: Custom processor function (dataset, **kwargs) -> TranslationDataset
        **kwargs: Additional arguments passed to processor
    
    Returns:
        TranslationDataset
    """
    # Default language mapping (English=0, French=1, German=2, etc.)
    if lang_map is None:
        lang_map = {'en': 0, 'fr': 1, 'de': 2, 'es': 3, 'it': 4, 'pt': 5, 'ru': 6, 'zh': 7, 'ja': 8}
    
    # Custom processor
    if processor_fn is not None:
        return processor_fn(dataset_name, lang_map=lang_map, split=split, **kwargs)
    
    # HuggingFace dataset
    if dataset_name.endswith('.jsonl'):
        return process_custom_jsonl(dataset_name, lang_map, **kwargs)
    else:
        # Load from HuggingFace
        dataset = load_dataset(dataset_name, lang_pair, **kwargs)
        return process_translation_dataset(dataset, lang_pair, lang_map, split)

