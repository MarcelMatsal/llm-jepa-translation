"""
Analyze CLS token embeddings across languages.

This script:
1. Loads a HuggingFace model
2. Extracts sentences from multiple languages
3. Passes them through the model to get CLS token embeddings
4. Performs PCA and t-SNE dimensionality reduction (2D and 3D)
5. Creates visualizations color-coded by language
6. Generates eigenspectrum plot showing variance explained

Usage:
    python scripts/analyze_cls_embeddings.py --model-name xlm-roberta-base
    python scripts/analyze_cls_embeddings.py --model-name maktzgls/bert-jepa --num-languages 15
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import List, Dict, Tuple
from tqdm import tqdm
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# Diverse set of languages to analyze
DEFAULT_LANGUAGES = [
    # European
    'de',  # German
    'fr',  # French
    'es',  # Spanish
    'it',  # Italian
    'nl',  # Dutch
    'sv',  # Swedish
    'pl',  # Polish
    'pt',  # Portuguese
    'ru',  # Russian
    # Asian
    'ja',  # Japanese
    'zh',  # Chinese
    'ko',  # Korean
    'ar',  # Arabic
    'hi',  # Hindi
    'vi',  # Vietnamese
]

# Language names for better visualization
LANGUAGE_NAMES = {
    'de': 'German', 'fr': 'French', 'es': 'Spanish', 'it': 'Italian',
    'nl': 'Dutch', 'sv': 'Swedish', 'pl': 'Polish', 'pt': 'Portuguese',
    'ru': 'Russian', 'ja': 'Japanese', 'zh': 'Chinese', 'ko': 'Korean',
    'ar': 'Arabic', 'hi': 'Hindi', 'vi': 'Vietnamese', 'en': 'English',
    'tr': 'Turkish', 'cs': 'Czech', 'fi': 'Finnish', 'no': 'Norwegian'
}


def collect_sentences_from_opus(
    languages: List[str],
    samples_per_language: int = 100,
    min_length: int = 20,
    max_length: int = 200
) -> Dict[str, List[str]]:
    """
    Collect sentences from OPUS-100 dataset for specified languages.
    
    Args:
        languages: List of language codes (e.g., ['de', 'fr', 'es'])
        samples_per_language: Number of sentences to collect per language
        min_length: Minimum sentence length in characters
        max_length: Maximum sentence length in characters
    
    Returns:
        Dictionary mapping language code to list of sentences
    """
    print("\n" + "="*80)
    print("Collecting Sentences from OPUS-100")
    print("="*80)
    
    sentences_by_lang = {}
    
    # We'll use English as the pivot language and extract both sides
    # This gives us en-XX and XX-en pairs
    for lang in tqdm(languages, desc="Loading languages"):
        if lang == 'en':
            continue  # Handle English separately
        
        sentences = []
        
        # Try both en-XX and XX-en configurations
        for lang_pair in [f'en-{lang}', f'{lang}-en']:
            try:
                dataset = load_dataset(
                    "Helsinki-NLP/opus-100",
                    lang_pair,
                    split='train',
                    streaming=True
                )
                
                # Determine which side is our target language
                lang1, lang2 = lang_pair.split('-')
                target_side = lang if lang == lang1 else lang2
                
                # Collect sentences
                for example in dataset:
                    if len(sentences) >= samples_per_language:
                        break
                    
                    translation = example.get('translation', {})
                    text = translation.get(target_side, '')
                    
                    # Filter by length
                    if min_length <= len(text) <= max_length:
                        sentences.append(text)
                
                if len(sentences) >= samples_per_language:
                    break
                    
            except Exception as e:
                continue
        
        if len(sentences) > 0:
            # Take only the requested number
            sentences_by_lang[lang] = sentences[:samples_per_language]
            print(f"  {lang}: {len(sentences_by_lang[lang])} sentences")
        else:
            print(f"  {lang}: Failed to load (skipping)")
    
    # Collect English sentences
    try:
        english_sentences = []
        dataset = load_dataset(
            "Helsinki-NLP/opus-100",
            "de-en",  # Use any pair with English
            split='train',
            streaming=True
        )
        
        for example in dataset:
            if len(english_sentences) >= samples_per_language:
                break
            
            translation = example.get('translation', {})
            text = translation.get('en', '')
            
            if min_length <= len(text) <= max_length:
                english_sentences.append(text)
        
        if len(english_sentences) > 0:
            sentences_by_lang['en'] = english_sentences
            print(f"  en: {len(sentences_by_lang['en'])} sentences")
    except Exception as e:
        print(f"  en: Failed to load (skipping)")
    
    print(f"\nTotal languages collected: {len(sentences_by_lang)}")
    print(f"Total sentences: {sum(len(s) for s in sentences_by_lang.values())}")
    
    return sentences_by_lang


def extract_cls_embeddings(
    model,
    tokenizer,
    sentences_by_lang: Dict[str, List[str]],
    device: str = 'cuda',
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract CLS token embeddings for all sentences.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        sentences_by_lang: Dictionary mapping language to sentences
        device: Device to run inference on
        batch_size: Batch size for inference
    
    Returns:
        embeddings: (N, hidden_dim) array of CLS embeddings
        labels: (N,) array of language code indices
        language_list: List of language codes in order
    """
    print("\n" + "="*80)
    print("Extracting CLS Token Embeddings")
    print("="*80)
    
    model.eval()
    model.to(device)
    
    all_embeddings = []
    all_labels = []
    language_list = sorted(sentences_by_lang.keys())
    
    with torch.no_grad():
        for lang_idx, lang in enumerate(tqdm(language_list, desc="Processing languages")):
            sentences = sentences_by_lang[lang]
            
            # Process in batches
            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i:i + batch_size]
                
                # Tokenize
                encoded = tokenizer(
                    batch_sentences,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Extract CLS token embeddings (first token, last layer)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                all_embeddings.append(cls_embeddings)
                all_labels.extend([lang_idx] * len(batch_sentences))
    
    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels)
    
    print(f"\nExtracted embeddings shape: {embeddings.shape}")
    print(f"Number of languages: {len(language_list)}")
    
    return embeddings, labels, language_list


def perform_pca(embeddings: np.ndarray, n_components: int = 50) -> Tuple[PCA, np.ndarray]:
    """
    Perform PCA on embeddings.
    
    Args:
        embeddings: (N, D) array of embeddings
        n_components: Number of components to compute
    
    Returns:
        pca: Fitted PCA object
        transformed: (N, n_components) transformed embeddings
    """
    print("\n" + "="*80)
    print("Performing PCA")
    print("="*80)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(embeddings)
    
    # Print variance explained
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print(f"\nVariance explained by first 10 components:")
    for i in range(min(10, n_components)):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.4f} "
              f"(cumulative: {cumulative_variance[i]:.4f})")
    
    return pca, transformed


def perform_tsne(embeddings: np.ndarray, n_components: int, random_state: int = 42) -> np.ndarray:
    """
    Perform t-SNE on embeddings.
    
    Args:
        embeddings: (N, D) array of embeddings
        n_components: 2 or 3 for 2D or 3D
        random_state: Random seed
    
    Returns:
        transformed: (N, n_components) transformed embeddings
    """
    print(f"\nPerforming t-SNE ({n_components}D)...")
    
    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=min(30, embeddings.shape[0] - 1),
        max_iter=1000,
        verbose=0
    )
    transformed = tsne.fit_transform(embeddings)
    
    return transformed


def plot_2d(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    language_list: List[str],
    title: str,
    output_path: str
):
    """Plot 2D embeddings with color coding by language."""
    plt.figure(figsize=(12, 8))
    
    # Create color palette
    n_languages = len(language_list)
    colors = sns.color_palette('husl', n_languages)
    
    # Plot each language
    for lang_idx, lang in enumerate(language_list):
        mask = labels == lang_idx
        lang_name = LANGUAGE_NAMES.get(lang, lang)
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[lang_idx]],
            label=lang_name,
            alpha=0.6,
            s=30
        )
    
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_3d(
    embeddings_3d: np.ndarray,
    labels: np.ndarray,
    language_list: List[str],
    title: str,
    output_path: str
):
    """Plot 3D embeddings with color coding by language."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create color palette
    n_languages = len(language_list)
    colors = sns.color_palette('husl', n_languages)
    
    # Plot each language
    for lang_idx, lang in enumerate(language_list):
        mask = labels == lang_idx
        lang_name = LANGUAGE_NAMES.get(lang, lang)
        ax.scatter(
            embeddings_3d[mask, 0],
            embeddings_3d[mask, 1],
            embeddings_3d[mask, 2],
            c=[colors[lang_idx]],
            label=lang_name,
            alpha=0.6,
            s=30
        )
    
    ax.set_xlabel('Component 1', fontsize=10)
    ax.set_ylabel('Component 2', fontsize=10)
    ax.set_zlabel('Component 3', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_eigenspectrum(pca: PCA, output_path: str, n_components: int = 50):
    """Plot eigenvalue spectrum showing variance explained by each component."""
    plt.figure(figsize=(14, 6))
    
    n_components = min(n_components, len(pca.explained_variance_))
    components = np.arange(1, n_components + 1)
    eigenvalues = pca.explained_variance_[:n_components]
    
    # Create bar chart
    plt.bar(components, eigenvalues, alpha=0.7, color='steelblue', edgecolor='black')
    
    plt.xlabel('Principal Component', fontsize=12)
    plt.ylabel('Eigenvalue (Variance)', fontsize=12)
    plt.title('Eigenspectrum: Variance Explained by Each Principal Component', 
              fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add cumulative variance line
    ax2 = plt.gca().twinx()
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_[:n_components])
    ax2.plot(components, cumulative_variance, 'r-', linewidth=2, marker='o', 
             markersize=4, label='Cumulative Variance')
    ax2.set_ylabel('Cumulative Variance Explained', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim([0, 1.05])
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def save_results(
    embeddings: np.ndarray,
    labels: np.ndarray,
    language_list: List[str],
    pca_2d: np.ndarray,
    pca_3d: np.ndarray,
    tsne_2d: np.ndarray,
    tsne_3d: np.ndarray,
    pca: PCA,
    output_dir: Path,
    model_name: str
):
    """Save embeddings and metadata."""
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)
    
    # Save embeddings
    np.savez(
        output_dir / 'embeddings.npz',
        embeddings=embeddings,
        labels=labels,
        pca_2d=pca_2d,
        pca_3d=pca_3d,
        tsne_2d=tsne_2d,
        tsne_3d=tsne_3d,
        eigenvalues=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_
    )
    print(f"  Saved: {output_dir / 'embeddings.npz'}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'language_list': language_list,
        'language_names': {lang: LANGUAGE_NAMES.get(lang, lang) for lang in language_list},
        'n_samples': int(embeddings.shape[0]),
        'embedding_dim': int(embeddings.shape[1]),
        'samples_per_language': {lang: int(np.sum(labels == i)) 
                                 for i, lang in enumerate(language_list)},
        'variance_explained_top10': pca.explained_variance_ratio_[:10].tolist(),
        'cumulative_variance_top10': np.cumsum(pca.explained_variance_ratio_[:10]).tolist()
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {output_dir / 'metadata.json'}")
    
    # Save summary
    with open(output_dir / 'analysis_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("CLS Token Embedding Analysis Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total samples: {embeddings.shape[0]}\n")
        f.write(f"Embedding dimension: {embeddings.shape[1]}\n")
        f.write(f"Number of languages: {len(language_list)}\n\n")
        
        f.write("Languages analyzed:\n")
        for i, lang in enumerate(language_list):
            lang_name = LANGUAGE_NAMES.get(lang, lang)
            n_samples = np.sum(labels == i)
            f.write(f"  {lang} ({lang_name}): {n_samples} samples\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("PCA Variance Explained\n")
        f.write("="*80 + "\n\n")
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        for i in range(min(20, len(pca.explained_variance_ratio_))):
            f.write(f"PC{i+1:2d}: {pca.explained_variance_ratio_[i]:7.4f} "
                   f"(cumulative: {cumulative[i]:.4f})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Generated Files\n")
        f.write("="*80 + "\n\n")
        f.write("Visualizations:\n")
        f.write("  - pca_2d.png: PCA projection in 2D\n")
        f.write("  - pca_3d.png: PCA projection in 3D\n")
        f.write("  - tsne_2d.png: t-SNE projection in 2D\n")
        f.write("  - tsne_3d.png: t-SNE projection in 3D\n")
        f.write("  - eigenspectrum.png: Eigenvalue spectrum\n\n")
        f.write("Data files:\n")
        f.write("  - embeddings.npz: Raw embeddings and projections\n")
        f.write("  - metadata.json: Analysis metadata\n")
        f.write("  - analysis_summary.txt: This file\n")
    
    print(f"  Saved: {output_dir / 'analysis_summary.txt'}")


def main(args):
    """Main analysis pipeline."""
    print("\n" + "="*80)
    print("CLS Token Embedding Analysis")
    print("="*80)
    print(f"\nModel: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Languages: {args.num_languages}")
    print(f"Samples per language: {args.samples_per_language}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory created: {output_dir}")
    
    # Select languages
    selected_languages = DEFAULT_LANGUAGES[:args.num_languages]
    print(f"\nSelected languages: {', '.join(selected_languages)}")
    
    # Step 1: Collect sentences
    sentences_by_lang = collect_sentences_from_opus(
        languages=selected_languages,
        samples_per_language=args.samples_per_language,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    if len(sentences_by_lang) == 0:
        print("\n❌ ERROR: No sentences collected!")
        return 1
    
    # Step 2: Load model
    print("\n" + "="*80)
    print("Loading Model")
    print("="*80)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name)
        print(f"✓ Model loaded: {args.model_name}")
        print(f"  Hidden size: {model.config.hidden_size}")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load model: {e}")
        return 1
    
    # Step 3: Extract CLS embeddings
    embeddings, labels, language_list = extract_cls_embeddings(
        model=model,
        tokenizer=tokenizer,
        sentences_by_lang=sentences_by_lang,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Step 4: PCA
    pca, pca_full = perform_pca(embeddings, n_components=50)
    pca_2d = pca_full[:, :2]
    pca_3d = pca_full[:, :3]
    
    # Step 5: t-SNE
    print("\n" + "="*80)
    print("Performing t-SNE")
    print("="*80)
    tsne_2d = perform_tsne(embeddings, n_components=2, random_state=args.seed)
    tsne_3d = perform_tsne(embeddings, n_components=3, random_state=args.seed)
    
    # Step 6: Create visualizations
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)
    
    plot_2d(
        pca_2d, labels, language_list,
        f'PCA 2D Projection of CLS Embeddings\nModel: {args.model_name}',
        output_dir / 'pca_2d.png'
    )
    
    plot_3d(
        pca_3d, labels, language_list,
        f'PCA 3D Projection of CLS Embeddings\nModel: {args.model_name}',
        output_dir / 'pca_3d.png'
    )
    
    plot_2d(
        tsne_2d, labels, language_list,
        f't-SNE 2D Projection of CLS Embeddings\nModel: {args.model_name}',
        output_dir / 'tsne_2d.png'
    )
    
    plot_3d(
        tsne_3d, labels, language_list,
        f't-SNE 3D Projection of CLS Embeddings\nModel: {args.model_name}',
        output_dir / 'tsne_3d.png'
    )
    
    plot_eigenspectrum(
        pca,
        output_dir / 'eigenspectrum.png',
        n_components=50
    )
    
    # Step 7: Save results
    save_results(
        embeddings=embeddings,
        labels=labels,
        language_list=language_list,
        pca_2d=pca_2d,
        pca_3d=pca_3d,
        tsne_2d=tsne_2d,
        tsne_3d=tsne_3d,
        pca=pca,
        output_dir=output_dir,
        model_name=args.model_name
    )
    
    # Final summary
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  Visualizations:")
    print("    - pca_2d.png")
    print("    - pca_3d.png")
    print("    - tsne_2d.png")
    print("    - tsne_3d.png")
    print("    - eigenspectrum.png")
    print("  Data:")
    print("    - embeddings.npz")
    print("    - metadata.json")
    print("    - analysis_summary.txt")
    print("\n✅ SUCCESS!")
    
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze CLS token embeddings across languages',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='HuggingFace model name or path (e.g., xlm-roberta-base, maktzgls/bert-jepa)'
    )
    
    parser.add_argument(
        '--num-languages',
        type=int,
        default=10,
        help='Number of languages to analyze'
    )
    
    parser.add_argument(
        '--samples-per-language',
        type=int,
        default=100,
        help='Number of sentences per language'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_output',
        help='Directory to save outputs'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--min-length',
        type=int,
        default=20,
        help='Minimum sentence length in characters'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=200,
        help='Maximum sentence length in characters'
    )
    
    args = parser.parse_args()
    
    exit(main(args))

