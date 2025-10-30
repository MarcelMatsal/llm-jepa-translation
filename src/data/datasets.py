"""
Dataset loader for WMT19 translation dataset.
"""
from typing import Dict, List
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset


class TranslationDataset(TorchDataset):
    """Dataset for translation pairs with language IDs."""
    
    def __init__(
        self,
        texts_src: List[str],
        texts_tgt: List[str],
        lang_src: List[int],
        lang_tgt: List[int]
    ):
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


def get_dataset(
    lang_pair: str,
    lang_map: Dict[str, int],
    split: str = 'train'
) -> TranslationDataset:
    """
    Load WMT19 translation dataset.
    
    Args:
        lang_pair: Language pair (e.g., 'en-de', 'cs-en')
        lang_map: Mapping from language codes to integer IDs
        split: Dataset split ('train' or 'validation')
    
    Returns:
        TranslationDataset
    """
    # Load dataset from HuggingFace
    dataset = load_dataset("wmt19", lang_pair)
    
    # Get split
    if split in dataset:
        split_dataset = dataset[split]
    else:
        raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")
    
    lang_src_code, lang_tgt_code = lang_pair.split('-')
    
    # Extract texts from translation dict
    # WMT19 format: {"translation": {"cs": "...", "en": "..."}}
    texts_src = []
    texts_tgt = []
    lang_src_ids = []
    lang_tgt_ids = []
    
    for example in split_dataset:
        translation = example['translation']
        
        if not isinstance(translation, dict):
            continue
        
        text_src = translation.get(lang_src_code, '')
        text_tgt = translation.get(lang_tgt_code, '')
        
        if text_src and text_tgt:
            texts_src.append(str(text_src))
            texts_tgt.append(str(text_tgt))
            lang_src_ids.append(lang_map.get(lang_src_code, 0))
            lang_tgt_ids.append(lang_map.get(lang_tgt_code, 1))
    
    if len(texts_src) == 0:
        raise ValueError(
            f"No valid samples found for lang_pair {lang_pair}. "
            f"Available languages in first example: {list(split_dataset[0]['translation'].keys()) if len(split_dataset) > 0 else 'empty dataset'}"
        )
    
    return TranslationDataset(texts_src, texts_tgt, lang_src_ids, lang_tgt_ids)
