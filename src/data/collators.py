"""
Data collation for batching.
"""
from torch.utils.data import DataLoader
from typing import Dict, List


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for batching translation pairs.
    
    Args:
        batch: List of dicts with 'text_src', 'text_tgt', 'lang_src', 'lang_tgt'
    
    Returns:
        Batched dictionary with same keys
    """
    texts_src = [item['text_src'] for item in batch]
    texts_tgt = [item['text_tgt'] for item in batch]
    lang_src = [item['lang_src'] for item in batch]
    lang_tgt = [item['lang_tgt'] for item in batch]
    
    return {
        'texts_src': texts_src,
        'texts_tgt': texts_tgt,
        'lang_src': lang_src,
        'lang_tgt': lang_tgt
    }


def get_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create DataLoader for dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

