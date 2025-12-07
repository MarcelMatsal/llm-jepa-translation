"""
Dataset loader for translation datasets with concatenated sequences.
Supports multiple language pairs for mixed batching.

Supported datasets:
- OPUS-100: Large parallel corpus with many language pairs
- Facebook FLORES: High-quality parallel corpus with 200+ languages (smaller but cleaner)

Also includes targeted fallbacks for language pairs missing in primary datasets.
"""
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
import random
import os
import tarfile
import urllib.request


# =============================================================================
# FLORES Language Code Mapping
# =============================================================================
# Maps our simple lang codes (e.g., 'de-en') to FLORES format (e.g., 'deu_Latn-eng_Latn')

FLORES_LANG_CODES = {
    # Language -> FLORES code
    'en': 'eng_Latn',
    'de': 'deu_Latn',
    'fr': 'fra_Latn',
    'es': 'spa_Latn',
    'el': 'ell_Grek',
    'bg': 'bul_Cyrl',
    'ru': 'rus_Cyrl',
    'tr': 'tur_Latn',
    'ar': 'arb_Arab',
    'vi': 'vie_Latn',
    'th': 'tha_Thai',
    'zh': 'zho_Hans',  # Simplified Chinese
    'hi': 'hin_Deva',
    'sw': 'swh_Latn',
    'ur': 'urd_Arab',
    # Add more as needed
}


def get_flores_config_name(lang_pair: str) -> str:
    """
    Convert our lang_pair format to FLORES config name.
    
    Args:
        lang_pair: e.g., 'de-en' or 'en-fr'
    
    Returns:
        FLORES config name, e.g., 'deu_Latn-eng_Latn'
    """
    lang1, lang2 = lang_pair.split('-')
    flores_lang1 = FLORES_LANG_CODES.get(lang1)
    flores_lang2 = FLORES_LANG_CODES.get(lang2)
    
    if flores_lang1 is None:
        raise ValueError(f"Unknown language code: {lang1}. Add it to FLORES_LANG_CODES.")
    if flores_lang2 is None:
        raise ValueError(f"Unknown language code: {lang2}. Add it to FLORES_LANG_CODES.")
    
    return f"{flores_lang1}-{flores_lang2}"


def get_flores_sentence_fields(lang_pair: str) -> Tuple[str, str]:
    """
    Get the FLORES sentence field names for a language pair.
    
    Args:
        lang_pair: e.g., 'de-en' or 'en-fr'
    
    Returns:
        Tuple of (field1, field2) e.g., ('sentence_deu_Latn', 'sentence_eng_Latn')
    """
    lang1, lang2 = lang_pair.split('-')
    flores_lang1 = FLORES_LANG_CODES.get(lang1)
    flores_lang2 = FLORES_LANG_CODES.get(lang2)
    
    return f"sentence_{flores_lang1}", f"sentence_{flores_lang2}"


# =============================================================================
# FLORES Direct Download (avoids trust_remote_code issues)
# =============================================================================
# The HuggingFace datasets library no longer supports trust_remote_code for
# datasets with custom loading scripts. We download directly from the official
# FLORES source, which is what the HF loading script does internally.

FLORES_URL = "https://tinyurl.com/flores200dataset"
FLORES_CACHE_DIR = os.path.expanduser("~/.cache/flores200")


def download_flores_data() -> str:
    """
    Download FLORES-200 dataset directly from official source.
    
    The dataset is a tar.gz archive that extracts to flores200_dataset/
    containing dev/ and devtest/ directories with sentence files.
    
    Returns:
        Path to the extracted dataset directory (flores200_dataset)
    """
    cache_dir = FLORES_CACHE_DIR
    dataset_dir = os.path.join(cache_dir, "flores200_dataset")
    
    # Check if already downloaded and extracted
    dev_dir = os.path.join(dataset_dir, "dev")
    devtest_dir = os.path.join(dataset_dir, "devtest")
    
    if os.path.exists(dev_dir) and os.path.exists(devtest_dir):
        # Verify at least one language file exists
        test_file = os.path.join(dev_dir, "eng_Latn.dev")
        if os.path.exists(test_file):
            return dataset_dir
    
    os.makedirs(cache_dir, exist_ok=True)
    
    print("  Downloading FLORES-200 dataset from official source...")
    archive_path = os.path.join(cache_dir, "flores200.tar.gz")
    
    # Download the archive
    try:
        urllib.request.urlretrieve(FLORES_URL, archive_path)
    except Exception as e:
        raise ValueError(f"Failed to download FLORES dataset: {e}")
    
    print("  Extracting FLORES-200 dataset...")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(cache_dir)
    except Exception as e:
        raise ValueError(f"Failed to extract FLORES dataset: {e}")
    
    # Clean up the archive to save space
    try:
        os.remove(archive_path)
    except:
        pass  # Not critical if cleanup fails
    
    # Verify extraction
    if not os.path.exists(dev_dir) or not os.path.exists(devtest_dir):
        raise ValueError(
            f"FLORES extraction failed: expected dev/ and devtest/ directories in {dataset_dir}"
        )
    
    print(f"  FLORES-200 cached at: {dataset_dir}")
    return dataset_dir


def load_flores_sentences(flores_dir: str, lang_code: str, split: str) -> List[str]:
    """
    Load sentences for a specific language and split from FLORES data.
    
    Args:
        flores_dir: Path to the flores200_dataset directory
        lang_code: FLORES language code (e.g., 'eng_Latn', 'deu_Latn')
        split: 'dev' or 'devtest'
    
    Returns:
        List of sentences (one per line in the file)
    """
    file_path = os.path.join(flores_dir, split, f"{lang_code}.{split}")
    
    if not os.path.exists(file_path):
        raise ValueError(f"FLORES file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]
    
    return sentences


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


def load_english_swahili_dataset(
    lang_pair: str,
    split: str = 'train',
    max_examples: Optional[int] = None,
    min_length: int = 10,
    max_length: int = 500
) -> TranslationPairDataset:
    """
    Load English–Swahili data from Rogendo/English-Swahili-Sentence-Pairs.

    The dataset only provides a train split, so any requested split will reuse it.
    """
    if split != 'train':
        print(f"  Note: Rogendo/English-Swahili-Sentence-Pairs only has 'train'; using it for split='{split}'.")

    try:
        dataset = load_dataset("Rogendo/English-Swahili-Sentence-Pairs", split="train")
    except Exception as e:
        raise ValueError(f"Failed to load Rogendo English-Swahili dataset: {e}")

    lang1_code, lang2_code = lang_pair.split('-')
    reverse_order = (lang1_code == 'sw' and lang2_code == 'en')

    total_available = len(dataset)
    num_to_load = min(max_examples, total_available) if max_examples else total_available
    print(f"  Available: {total_available} examples, Loading: {num_to_load}")

    examples = []
    for example in dataset:
        english_text = example.get('English sentence', '')
        swahili_text = example.get('Swahili Translation', '')

        text1, text2 = (swahili_text, english_text) if reverse_order else (english_text, swahili_text)

        if (
            text1 and text2 and
            len(text1) >= min_length and len(text2) >= min_length and
            len(text1) <= max_length and len(text2) <= max_length
        ):
            examples.append({'text1': str(text1), 'text2': str(text2)})

        if len(examples) >= num_to_load:
            break

    if len(examples) == 0:
        raise ValueError(
            "No valid English-Swahili examples found. "
            "Check that length filters are appropriate."
        )

    return TranslationPairDataset(examples, lang_pair)


# Map of language pairs to custom dataset loader functions (non-OPUS sources)
CUSTOM_DATASET_LOADERS = {
    'en-sw': load_english_swahili_dataset,
    'sw-en': load_english_swahili_dataset,
}


def load_flores_dataset(
    lang_pair: str,
    split: str = 'train',
    max_examples: Optional[int] = None,
    min_length: int = 10,
    max_length: int = 500
) -> TranslationPairDataset:
    """
    Load Facebook FLORES dataset for a single language pair.
    
    Downloads directly from the official FLORES source to avoid HuggingFace
    trust_remote_code issues.
    
    FLORES only has 'dev' (997) and 'devtest' (1012) splits natively.
    We load ALL data and create our own train/validation/test splits
    to match the OPUS-100 split style:
    - 'train' -> 80% of all data (~1607 examples)
    - 'validation' -> 10% of all data (~201 examples)
    - 'test' -> 10% of all data (~201 examples)
    - 'all' -> 100% of all data (~2009 examples)
    
    The split is deterministic (uses fixed seed) for reproducibility.
    
    Args:
        lang_pair: Language pair (e.g., 'de-en', 'en-fr')
        split: Dataset split ('train', 'validation', 'test', or 'all')
        max_examples: Maximum number of examples to load
        min_length: Minimum text length (characters)
        max_length: Maximum text length (characters)
    
    Returns:
        TranslationPairDataset
    """
    # Get FLORES language codes
    lang1, lang2 = lang_pair.split('-')
    flores_lang1 = FLORES_LANG_CODES.get(lang1)
    flores_lang2 = FLORES_LANG_CODES.get(lang2)
    
    if flores_lang1 is None:
        raise ValueError(f"Unknown language code: {lang1}. Add it to FLORES_LANG_CODES.")
    if flores_lang2 is None:
        raise ValueError(f"Unknown language code: {lang2}. Add it to FLORES_LANG_CODES.")
    
    # Validate split
    valid_splits = ('train', 'validation', 'test', 'all')
    if split not in valid_splits:
        raise ValueError(f"Unknown split: {split}. Use one of {valid_splits}.")
    
    print(f"  Loading FLORES: {flores_lang1}-{flores_lang2}, split: {split}")
    
    # Download/cache FLORES data
    flores_dir = download_flores_data()
    
    # Always load ALL data (dev + devtest)
    all_sentences_lang1 = []
    all_sentences_lang2 = []
    
    for flores_split in ['dev', 'devtest']:
        try:
            sentences_lang1 = load_flores_sentences(flores_dir, flores_lang1, flores_split)
            sentences_lang2 = load_flores_sentences(flores_dir, flores_lang2, flores_split)
            
            if len(sentences_lang1) != len(sentences_lang2):
                raise ValueError(
                    f"Sentence count mismatch for {flores_split}: "
                    f"{flores_lang1}={len(sentences_lang1)}, {flores_lang2}={len(sentences_lang2)}"
                )
            
            all_sentences_lang1.extend(sentences_lang1)
            all_sentences_lang2.extend(sentences_lang2)
            
        except Exception as e:
            raise ValueError(f"Failed to load FLORES for {lang_pair}: {e}")
    
    # Filter by length first (before splitting)
    filtered_pairs = []
    for text1, text2 in zip(all_sentences_lang1, all_sentences_lang2):
        if (text1 and text2 and 
            len(text1) >= min_length and len(text2) >= min_length and
            len(text1) <= max_length and len(text2) <= max_length):
            filtered_pairs.append({'text1': str(text1), 'text2': str(text2)})
    
    if len(filtered_pairs) == 0:
        raise ValueError(
            f"No valid examples found for {lang_pair} in FLORES. "
            f"Check that the language codes are correct and length filters are appropriate."
        )
    
    # Create deterministic split (shuffle with fixed seed, then slice)
    # Use a fixed seed for reproducibility across runs
    rng = random.Random(42)
    indices = list(range(len(filtered_pairs)))
    rng.shuffle(indices)
    
    # Split ratios: 80% train, 10% validation, 10% test
    n_total = len(indices)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    # n_test = n_total - n_train - n_val (remaining)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Select indices based on requested split
    if split == 'train':
        selected_indices = train_indices
    elif split == 'validation':
        selected_indices = val_indices
    elif split == 'test':
        selected_indices = test_indices
    elif split == 'all':
        selected_indices = indices  # All data
    
    # Extract examples for this split
    examples = [filtered_pairs[i] for i in selected_indices]
    
    # Apply max_examples limit if specified
    if max_examples is not None and len(examples) > max_examples:
        examples = examples[:max_examples]
    
    print(f"  Total FLORES data: {n_total}, Split '{split}': {len(examples)} examples")
    print(f"    (train: {n_train}, validation: {n_val}, test: {len(test_indices)})")
    
    return TranslationPairDataset(examples, lang_pair)


def load_opus100_dataset(
    lang_pair: str,
    split: str = 'train',
    max_examples: Optional[int] = None,
    min_length: int = 10,
    max_length: int = 500
) -> TranslationPairDataset:
    """
    Load OPUS-100 translation dataset for a single language pair.
    
    Args:
        lang_pair: Language pair (e.g., 'de-en', 'cs-en', 'en-fr')
        split: Dataset split ('train', 'validation', or 'test')
        max_examples: Maximum number of examples to load
        min_length: Minimum text length (characters)
        max_length: Maximum text length (characters)
    
    Returns:
        TranslationPairDataset
    """
    # Use custom loaders when OPUS-100 does not provide the language pair
    custom_loader = CUSTOM_DATASET_LOADERS.get(lang_pair)
    if custom_loader is not None:
        print(f"Loading {lang_pair} from custom dataset...")
        return custom_loader(
            lang_pair=lang_pair,
            split=split,
            max_examples=max_examples,
            min_length=min_length,
            max_length=max_length
        )

    # Load dataset from HuggingFace OPUS-100
    try:
        dataset = load_dataset("Helsinki-NLP/opus-100", lang_pair, split=split)
    except Exception as e:
        raise ValueError(f"Failed to load OPUS-100 dataset for {lang_pair}: {e}")
    
    lang1_code, lang2_code = lang_pair.split('-')
    
    # Determine the actual number of examples to process
    total_available = len(dataset)
    num_to_load = min(max_examples, total_available) if max_examples else total_available
    
    print(f"  Available: {total_available} examples, Loading: {num_to_load}")
    
    # Extract and filter examples
    examples = []
    for i, example in enumerate(dataset):
        # Stop once we've collected enough valid examples
        if len(examples) >= num_to_load:
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
            f"Check that the language codes are correct and length filters are appropriate."
        )
    
    return TranslationPairDataset(examples, lang_pair)


def load_translation_dataset(
    lang_pair: str,
    split: str = 'train',
    max_examples: Optional[int] = None,
    min_length: int = 10,
    max_length: int = 500,
    dataset_source: str = 'opus100'
) -> TranslationPairDataset:
    """
    Load translation dataset for a single language pair.
    
    Args:
        lang_pair: Language pair (e.g., 'de-en', 'cs-en', 'en-fr')
        split: Dataset split ('train', 'validation', 'test', or 'all' for FLORES)
        max_examples: Maximum number of examples to load. 
                     If None, loads all available examples.
                     If specified, loads min(max_examples, total_available).
        min_length: Minimum text length (characters)
        max_length: Maximum text length (characters)
        dataset_source: Which dataset to use ('opus100' or 'flores')
    
    Returns:
        TranslationPairDataset
    """
    if dataset_source == 'flores':
        return load_flores_dataset(
            lang_pair=lang_pair,
            split=split,
            max_examples=max_examples,
            min_length=min_length,
            max_length=max_length
        )
    elif dataset_source == 'opus100':
        return load_opus100_dataset(
            lang_pair=lang_pair,
            split=split,
            max_examples=max_examples,
            min_length=min_length,
            max_length=max_length
        )
    else:
        raise ValueError(f"Unknown dataset_source: {dataset_source}. Use 'opus100' or 'flores'.")


def load_multilingual_dataset(
    lang_pairs: List[str],
    split: str = 'train',
    max_examples_per_pair: Optional[int] = None,
    min_length: int = 10,
    max_length: int = 500,
    dataset_source: str = 'opus100'
) -> MultilingualDataset:
    """
    Load multiple language pairs into a single mixed dataset.
    
    Args:
        lang_pairs: List of language pairs (e.g., ['de-en', 'fr-en', 'en-es'])
        split: Dataset split ('train', 'validation', 'test', or 'all' for FLORES)
        max_examples_per_pair: Maximum examples per language pair.
                               If None, loads all available examples for each pair.
                               If specified, loads min(max_examples_per_pair, total_available)
                               for each language pair.
        min_length: Minimum text length (characters)
        max_length: Maximum text length (characters)
        dataset_source: Which dataset to use ('opus100' or 'flores')
    
    Returns:
        MultilingualDataset with mixed language pairs
    """
    datasets = []
    
    print(f"Using dataset source: {dataset_source}")
    
    for lang_pair in lang_pairs:
        print(f"Loading {lang_pair}...")
        try:
            dataset = load_translation_dataset(
                lang_pair=lang_pair,
                split=split,
                max_examples=max_examples_per_pair,
                min_length=min_length,
                max_length=max_length,
                dataset_source=dataset_source
            )
            datasets.append(dataset)
            print(f"  Loaded {len(dataset)} examples")
        except Exception as e:
            print(f"  Warning: Failed to load {lang_pair}: {e}")
            continue
    
    if len(datasets) == 0:
        raise ValueError("No datasets loaded successfully")
    
    return MultilingualDataset(datasets)
