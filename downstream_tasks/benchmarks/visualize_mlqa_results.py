#!/usr/bin/env python3
"""
Visualize MLQA benchmark results across multiple models.
Creates individual and comparison plots for XLT and G-XLT performance.

Usage:
    python visualize_mlqa_results.py

Outputs plots to: downstream_tasks/benchmarks/plots/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Configure matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# MLQA Languages
LANGUAGES = ['en', 'ar', 'de', 'es', 'hi', 'vi', 'zh']
LANG_NAMES = {
    'en': 'English', 'ar': 'Arabic', 'de': 'German', 
    'es': 'Spanish', 'hi': 'Hindi', 'vi': 'Vietnamese', 'zh': 'Chinese'
}

# Model results - parsed from the log files
RESULTS = {
    'XLM-RoBERTa Base': {
        'short_name': 'XLM-R',
        'f1_matrix': {
            'en': {'en': 80.9, 'ar': 31.0, 'de': 60.6, 'es': 59.3, 'hi': 42.1, 'vi': 42.6, 'zh': 37.2},
            'ar': {'en': 51.9, 'ar': 55.0, 'de': 35.2, 'es': 29.2, 'hi': 17.1, 'vi': 15.9, 'zh': 17.4},
            'de': {'en': 61.2, 'ar': 22.8, 'de': 62.4, 'es': 46.4, 'hi': 27.7, 'vi': 32.0, 'zh': 30.1},
            'es': {'en': 66.2, 'ar': 22.1, 'de': 51.0, 'es': 66.8, 'hi': 27.9, 'vi': 32.3, 'zh': 29.1},
            'hi': {'en': 60.0, 'ar': 22.0, 'de': 45.0, 'es': 39.4, 'hi': 61.7, 'vi': 27.8, 'zh': 30.1},
            'vi': {'en': 63.7, 'ar': 17.3, 'de': 41.8, 'es': 35.9, 'hi': 26.9, 'vi': 67.4, 'zh': 27.4},
            'zh': {'en': 36.2, 'ar': 4.6, 'de': 16.6, 'es': 13.1, 'hi': 10.0, 'vi': 8.6, 'zh': 40.3},
        },
        'em_matrix': {
            'en': {'en': 68.2, 'ar': 19.2, 'de': 46.0, 'es': 43.0, 'hi': 29.1, 'vi': 29.0, 'zh': 24.7},
            'ar': {'en': 34.3, 'ar': 35.9, 'de': 21.6, 'es': 16.0, 'hi': 7.6, 'vi': 7.5, 'zh': 7.8},
            'de': {'en': 46.2, 'ar': 13.8, 'de': 46.8, 'es': 33.0, 'hi': 18.0, 'vi': 19.2, 'zh': 18.4},
            'es': {'en': 45.7, 'ar': 11.0, 'de': 32.7, 'es': 45.8, 'hi': 16.0, 'vi': 18.3, 'zh': 16.4},
            'hi': {'en': 43.2, 'ar': 11.4, 'de': 31.7, 'es': 25.1, 'hi': 44.0, 'vi': 15.5, 'zh': 18.2},
            'vi': {'en': 44.7, 'ar': 7.6, 'de': 26.3, 'es': 21.7, 'hi': 14.2, 'vi': 47.0, 'zh': 15.1},
            'zh': {'en': 35.5, 'ar': 4.3, 'de': 15.9, 'es': 12.5, 'hi': 9.6, 'vi': 8.0, 'zh': 39.6},
        }
    },
    'JEPA Bilingual (15-lang)': {
        'short_name': 'JEPA-Bi',
        'f1_matrix': {
            'en': {'en': 81.2, 'ar': 60.7, 'de': 69.9, 'es': 70.2, 'hi': 66.9, 'vi': 64.7, 'zh': 67.0},
            'ar': {'en': 54.7, 'ar': 55.9, 'de': 49.0, 'es': 46.5, 'hi': 38.0, 'vi': 40.5, 'zh': 43.7},
            'de': {'en': 62.8, 'ar': 48.1, 'de': 63.3, 'es': 51.5, 'hi': 50.0, 'vi': 48.2, 'zh': 53.9},
            'es': {'en': 68.2, 'ar': 52.5, 'de': 61.2, 'es': 67.8, 'hi': 53.5, 'vi': 52.0, 'zh': 58.1},
            'hi': {'en': 62.3, 'ar': 39.0, 'de': 53.0, 'es': 48.3, 'hi': 63.2, 'vi': 47.7, 'zh': 50.5},
            'vi': {'en': 66.4, 'ar': 43.0, 'de': 51.8, 'es': 42.6, 'hi': 49.8, 'vi': 68.5, 'zh': 56.1},
            'zh': {'en': 39.6, 'ar': 21.9, 'de': 30.0, 'es': 27.4, 'hi': 27.9, 'vi': 28.5, 'zh': 42.3},
        },
        'em_matrix': {
            'en': {'en': 68.2, 'ar': 45.4, 'de': 55.8, 'es': 54.1, 'hi': 51.3, 'vi': 49.1, 'zh': 50.8},
            'ar': {'en': 36.7, 'ar': 37.0, 'de': 33.5, 'es': 29.8, 'hi': 22.8, 'vi': 25.3, 'zh': 27.5},
            'de': {'en': 46.7, 'ar': 35.4, 'de': 47.2, 'es': 36.7, 'hi': 36.2, 'vi': 33.9, 'zh': 37.8},
            'es': {'en': 47.1, 'ar': 33.1, 'de': 41.4, 'es': 46.4, 'hi': 35.2, 'vi': 34.0, 'zh': 37.8},
            'hi': {'en': 45.3, 'ar': 25.3, 'de': 37.8, 'es': 32.9, 'hi': 45.4, 'vi': 32.4, 'zh': 34.9},
            'vi': {'en': 46.6, 'ar': 27.7, 'de': 34.4, 'es': 29.0, 'hi': 33.8, 'vi': 47.8, 'zh': 38.2},
            'zh': {'en': 38.9, 'ar': 21.5, 'de': 29.1, 'es': 26.8, 'hi': 27.3, 'vi': 27.8, 'zh': 41.6},
        }
    },
    'JEPA Monolingual (15-lang)': {
        'short_name': 'JEPA-Mono',
        'f1_matrix': {
            'en': {'en': 81.0, 'ar': 49.3, 'de': 64.7, 'es': 65.6, 'hi': 57.5, 'vi': 58.6, 'zh': 54.9},
            'ar': {'en': 52.0, 'ar': 55.5, 'de': 44.0, 'es': 42.2, 'hi': 26.4, 'vi': 32.0, 'zh': 31.5},
            'de': {'en': 60.3, 'ar': 38.5, 'de': 62.4, 'es': 50.9, 'hi': 42.9, 'vi': 44.2, 'zh': 44.7},
            'es': {'en': 66.2, 'ar': 44.8, 'de': 58.0, 'es': 67.8, 'hi': 45.2, 'vi': 49.3, 'zh': 48.7},
            'hi': {'en': 61.3, 'ar': 37.1, 'de': 52.3, 'es': 46.3, 'hi': 64.0, 'vi': 46.2, 'zh': 44.4},
            'vi': {'en': 63.9, 'ar': 38.3, 'de': 49.6, 'es': 39.9, 'hi': 45.1, 'vi': 68.6, 'zh': 46.8},
            'zh': {'en': 36.6, 'ar': 14.3, 'de': 24.0, 'es': 21.7, 'hi': 22.4, 'vi': 22.3, 'zh': 41.3},
        },
        'em_matrix': {
            'en': {'en': 68.0, 'ar': 34.6, 'de': 50.1, 'es': 49.2, 'hi': 42.8, 'vi': 43.3, 'zh': 39.9},
            'ar': {'en': 34.7, 'ar': 36.6, 'de': 28.8, 'es': 25.6, 'hi': 15.2, 'vi': 18.9, 'zh': 18.1},
            'de': {'en': 44.9, 'ar': 26.6, 'de': 46.4, 'es': 35.6, 'hi': 30.1, 'vi': 30.2, 'zh': 31.6},
            'es': {'en': 45.3, 'ar': 27.3, 'de': 38.5, 'es': 46.1, 'hi': 28.7, 'vi': 31.2, 'zh': 30.7},
            'hi': {'en': 44.1, 'ar': 23.0, 'de': 36.7, 'es': 30.4, 'hi': 46.0, 'vi': 30.6, 'zh': 29.5},
            'vi': {'en': 44.1, 'ar': 23.0, 'de': 31.8, 'es': 25.4, 'hi': 28.9, 'vi': 47.6, 'zh': 30.2},
            'zh': {'en': 35.9, 'ar': 14.0, 'de': 23.1, 'es': 21.0, 'hi': 22.1, 'vi': 21.7, 'zh': 40.6},
        }
    }
}


def matrix_to_df(matrix_dict):
    """Convert nested dict matrix to DataFrame."""
    df = pd.DataFrame(matrix_dict).T
    df = df[LANGUAGES]  # Ensure column order
    df = df.reindex(LANGUAGES)  # Ensure row order
    return df


def compute_statistics(model_data):
    """Compute XLT and G-XLT averages for a model."""
    f1_df = matrix_to_df(model_data['f1_matrix'])
    em_df = matrix_to_df(model_data['em_matrix'])
    
    # XLT = diagonal (same language)
    xlt_f1 = np.diag(f1_df.values).mean()
    xlt_em = np.diag(em_df.values).mean()
    
    # G-XLT = off-diagonal (cross-lingual)
    mask = ~np.eye(len(LANGUAGES), dtype=bool)
    gxlt_f1 = f1_df.values[mask].mean()
    gxlt_em = em_df.values[mask].mean()
    
    # Overall
    overall_f1 = f1_df.values.mean()
    overall_em = em_df.values.mean()
    
    return {
        'xlt_f1': xlt_f1, 'xlt_em': xlt_em,
        'gxlt_f1': gxlt_f1, 'gxlt_em': gxlt_em,
        'overall_f1': overall_f1, 'overall_em': overall_em
    }


def plot_heatmap(matrix_df, title, output_path, cmap='RdYlGn', vmin=0, vmax=100, metric='F1'):
    """Create a heatmap for a G-XLT matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        matrix_df, 
        annot=True, 
        fmt='.1f', 
        cmap=cmap,
        vmin=vmin, 
        vmax=vmax,
        ax=ax,
        cbar_kws={'label': f'{metric} Score'},
        annot_kws={'size': 11}
    )
    
    # Customize labels
    ax.set_xlabel('Question Language', fontsize=12, fontweight='bold')
    ax.set_ylabel('Context Language', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Use full language names
    ax.set_xticklabels([LANG_NAMES[l] for l in LANGUAGES], rotation=45, ha='right')
    ax.set_yticklabels([LANG_NAMES[l] for l in LANGUAGES], rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_xlt_comparison(stats_dict, output_path):
    """Create bar chart comparing XLT (same-language) performance."""
    models = list(stats_dict.keys())
    short_names = [RESULTS[m]['short_name'] for m in models]
    f1_scores = [stats_dict[m]['xlt_f1'] for m in models]
    em_scores = [stats_dict[m]['xlt_em'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, em_scores, width, label='Exact Match', color='#3498db', edgecolor='black')
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('XLT Performance (Same-Language QA)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names)
    ax.legend()
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_gxlt_comparison(stats_dict, output_path):
    """Create bar chart comparing G-XLT (cross-lingual) performance."""
    models = list(stats_dict.keys())
    short_names = [RESULTS[m]['short_name'] for m in models]
    f1_scores = [stats_dict[m]['gxlt_f1'] for m in models]
    em_scores = [stats_dict[m]['gxlt_em'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score', color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x + width/2, em_scores, width, label='Exact Match', color='#9b59b6', edgecolor='black')
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('G-XLT Performance (Cross-Lingual QA)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names)
    ax.legend()
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_per_language_comparison(output_path):
    """Create grouped bar chart comparing per-language XLT F1 scores."""
    models = list(RESULTS.keys())
    short_names = [RESULTS[m]['short_name'] for m in models]
    
    x = np.arange(len(LANGUAGES))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for i, (model, color) in enumerate(zip(models, colors)):
        f1_scores = [RESULTS[model]['f1_matrix'][lang][lang] for lang in LANGUAGES]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, f1_scores, width, label=RESULTS[model]['short_name'], 
                     color=color, edgecolor='black')
    
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('XLT F1 Score by Language', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[l] for l in LANGUAGES])
    ax.legend()
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_improvement_heatmap(output_path):
    """Create heatmap showing F1 improvement of JEPA Bilingual over XLM-RoBERTa."""
    xlmr_df = matrix_to_df(RESULTS['XLM-RoBERTa Base']['f1_matrix'])
    jepa_bi_df = matrix_to_df(RESULTS['JEPA Bilingual (15-lang)']['f1_matrix'])
    
    diff_df = jepa_bi_df - xlmr_df
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use diverging colormap centered at 0
    max_val = max(abs(diff_df.values.min()), abs(diff_df.values.max()))
    
    sns.heatmap(
        diff_df, 
        annot=True, 
        fmt='+.1f', 
        cmap='RdYlGn',
        center=0,
        vmin=-max_val,
        vmax=max_val,
        ax=ax,
        cbar_kws={'label': 'F1 Improvement'},
        annot_kws={'size': 11}
    )
    
    ax.set_xlabel('Question Language', fontsize=12, fontweight='bold')
    ax.set_ylabel('Context Language', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score Improvement: JEPA Bilingual vs XLM-RoBERTa', fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xticklabels([LANG_NAMES[l] for l in LANGUAGES], rotation=45, ha='right')
    ax.set_yticklabels([LANG_NAMES[l] for l in LANGUAGES], rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_summary_comparison(stats_dict, output_path):
    """Create comprehensive summary comparison chart."""
    models = list(stats_dict.keys())
    short_names = [RESULTS[m]['short_name'] for m in models]
    
    metrics = ['XLT F1', 'XLT EM', 'G-XLT F1', 'G-XLT EM', 'Overall F1', 'Overall EM']
    
    data = []
    for model in models:
        s = stats_dict[model]
        data.append([s['xlt_f1'], s['xlt_em'], s['gxlt_f1'], s['gxlt_em'], s['overall_f1'], s['overall_em']])
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for i, (model_data, color, name) in enumerate(zip(data, colors, short_names)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, model_data, width, label=name, color=color, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('MLQA Benchmark Summary: All Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    
    # Add vertical separators
    for i in [1.5, 3.5]:
        ax.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_radar_comparison(stats_dict, output_path):
    """Create radar chart comparing models across metrics."""
    from math import pi
    
    models = list(stats_dict.keys())
    short_names = [RESULTS[m]['short_name'] for m in models]
    
    # Categories
    categories = ['XLT F1', 'XLT EM', 'G-XLT F1', 'G-XLT EM']
    N = len(categories)
    
    # Create angle for each category
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for model, color in zip(models, colors):
        s = stats_dict[model]
        values = [s['xlt_f1'], s['xlt_em'], s['gxlt_f1'], s['gxlt_em']]
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=RESULTS[model]['short_name'], color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 80)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Model Comparison Across Metrics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_paper_table(output_path=None):
    """Generate a table in the MLQA paper format (Table 3 style)."""
    
    # Paper baselines (from MLQA paper - Lewis et al., 2019)
    PAPER_BASELINES = {
        'BERT-Large†': {'train': 'en', 'lgs': 1, 'en': (80.2, 67.4), 'es': None, 'de': None, 'ar': None, 'hi': None, 'vi': None, 'zh': None},
        'mBERT†': {'train': 'en', 'lgs': 102, 'en': (77.7, 65.2), 'es': (64.3, 46.6), 'de': (57.9, 44.3), 'ar': (45.7, 29.8), 'hi': (43.8, 29.7), 'vi': (57.1, 38.6), 'zh': (57.5, 37.3)},
        'XLM-15†': {'train': 'en', 'lgs': 15, 'en': (74.9, 62.4), 'es': (68.0, 49.8), 'de': (62.2, 47.6), 'ar': (54.8, 36.3), 'hi': (48.8, 27.3), 'vi': (61.4, 41.8), 'zh': (61.1, 39.6)},
        'XLM-R_Base': {'train': 'en', 'lgs': 100, 'en': (77.1, 64.6), 'es': (67.4, 49.6), 'de': (60.9, 46.7), 'ar': (54.9, 36.6), 'hi': (59.4, 42.9), 'vi': (64.5, 44.7), 'zh': (61.8, 39.3)},
        'XLM-R': {'train': 'en', 'lgs': 100, 'en': (80.6, 67.8), 'es': (74.1, 56.0), 'de': (68.5, 53.6), 'ar': (63.1, 43.5), 'hi': (69.2, 51.6), 'vi': (71.3, 50.9), 'zh': (68.0, 45.4)},
    }
    
    # Our results (XLT - diagonal values from matrices)
    OUR_RESULTS = {
        'XLM-R Base (Ours)': {
            'train': 'en', 'lgs': 100,
            'en': (80.9, 68.2), 'es': (66.8, 45.8), 'de': (62.4, 46.8), 
            'ar': (55.0, 35.9), 'hi': (61.7, 44.0), 'vi': (67.4, 47.0), 'zh': (40.3, 39.6)
        },
        'JEPA-Mono (Ours)': {
            'train': 'en', 'lgs': 15,
            'en': (81.0, 68.0), 'es': (67.8, 46.1), 'de': (62.4, 46.4),
            'ar': (55.5, 36.6), 'hi': (64.0, 46.0), 'vi': (68.6, 47.6), 'zh': (41.3, 40.6)
        },
        'JEPA-Bi (Ours)': {
            'train': 'en', 'lgs': 15,
            'en': (81.2, 68.2), 'es': (67.8, 46.4), 'de': (63.3, 47.2),
            'ar': (55.9, 37.0), 'hi': (63.2, 45.4), 'vi': (68.5, 47.8), 'zh': (42.3, 41.6)
        },
    }
    
    # Combine all results
    all_results = {**PAPER_BASELINES, **OUR_RESULTS}
    
    # Language order as in paper
    lang_order = ['en', 'es', 'de', 'ar', 'hi', 'vi', 'zh']
    
    # Build table
    lines = []
    
    # Header
    header = f"{'Model':<20} {'train':>6} {'#lgs':>5}"
    for lang in lang_order:
        header += f" {lang:>12}"
    header += f" {'Avg':>12}"
    
    lines.append("=" * len(header))
    lines.append("Table: Results on MLQA Question Answering (XLT Task)")
    lines.append("F1 / EM scores for zero-shot classification")
    lines.append("Models fine-tuned on English SQuAD, evaluated on 7 languages")
    lines.append("=" * len(header))
    lines.append("")
    lines.append(header)
    lines.append("-" * len(header))
    
    # Data rows
    for model_name, data in all_results.items():
        row = f"{model_name:<20} {data['train']:>6} {data['lgs']:>5}"
        
        f1_sum, em_sum, count = 0, 0, 0
        for lang in lang_order:
            if data[lang] is not None:
                f1, em = data[lang]
                row += f" {f1:>5.1f}/{em:>4.1f}"
                f1_sum += f1
                em_sum += em
                count += 1
            else:
                row += f" {'-':>12}"
        
        # Average
        if count > 0:
            avg_f1 = f1_sum / count
            avg_em = em_sum / count
            row += f" {avg_f1:>5.1f}/{avg_em:>4.1f}"
        else:
            row += f" {'-':>12}"
        
        lines.append(row)
        
        # Add separator after paper baselines
        if model_name == 'XLM-R':
            lines.append("-" * len(header))
    
    lines.append("=" * len(header))
    lines.append("")
    lines.append("† Results from original MLQA paper (Lewis et al., 2019)")
    lines.append("(Ours) = Our experimental results")
    
    table_str = "\n".join(lines)
    
    # Print to console
    print("\n" + table_str)
    
    # Save to file
    if output_path:
        with open(output_path, 'w') as f:
            f.write(table_str)
        print(f"\nTable saved to: {output_path}")
    
    return table_str


def generate_latex_table(output_path=None):
    """Generate LaTeX table in paper format."""
    
    # Our results (XLT - diagonal values)
    results = {
        'XLM-R Base (Ours)': {
            'train': 'en', 'lgs': 100,
            'en': (80.9, 68.2), 'es': (66.8, 45.8), 'de': (62.4, 46.8), 
            'ar': (55.0, 35.9), 'hi': (61.7, 44.0), 'vi': (67.4, 47.0), 'zh': (40.3, 39.6)
        },
        'JEPA-Mono': {
            'train': 'en', 'lgs': 15,
            'en': (81.0, 68.0), 'es': (67.8, 46.1), 'de': (62.4, 46.4),
            'ar': (55.5, 36.6), 'hi': (64.0, 46.0), 'vi': (68.6, 47.6), 'zh': (41.3, 40.6)
        },
        'JEPA-Bi': {
            'train': 'en', 'lgs': 15,
            'en': (81.2, 68.2), 'es': (67.8, 46.4), 'de': (63.3, 47.2),
            'ar': (55.9, 37.0), 'hi': (63.2, 45.4), 'vi': (68.5, 47.8), 'zh': (42.3, 41.6)
        },
    }
    
    lang_order = ['en', 'es', 'de', 'ar', 'hi', 'vi', 'zh']
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcc" + "c" * len(lang_order) + "c}")
    lines.append(r"\toprule")
    
    # Header
    header = r"\textbf{Model} & \textbf{train} & \textbf{\#lgs}"
    for lang in lang_order:
        header += f" & \\textbf{{{lang}}}"
    header += r" & \textbf{Avg} \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # Paper baselines
    lines.append(r"BERT-Large$^\dagger$ & en & 1 & 80.2/67.4 & - & - & - & - & - & - & - \\")
    lines.append(r"mBERT$^\dagger$ & en & 102 & 77.7/65.2 & 64.3/46.6 & 57.9/44.3 & 45.7/29.8 & 43.8/29.7 & 57.1/38.6 & 57.5/37.3 & 57.7/41.6 \\")
    lines.append(r"XLM-15$^\dagger$ & en & 15 & 74.9/62.4 & 68.0/49.8 & 62.2/47.6 & 54.8/36.3 & 48.8/27.3 & 61.4/41.8 & 61.1/39.6 & 61.6/43.5 \\")
    lines.append(r"XLM-R$_{\text{Base}}$ & en & 100 & 77.1/64.6 & 67.4/49.6 & 60.9/46.7 & 54.9/36.6 & 59.4/42.9 & 64.5/44.7 & 61.8/39.3 & 63.7/46.3 \\")
    lines.append(r"XLM-R & en & 100 & 80.6/67.8 & 74.1/56.0 & 68.5/53.6 & 63.1/43.5 & 69.2/51.6 & 71.3/50.9 & 68.0/45.4 & 70.7/52.7 \\")
    lines.append(r"\midrule")
    
    # Our results
    for model_name, data in results.items():
        row = f"{model_name} & {data['train']} & {data['lgs']}"
        
        f1_sum, em_sum = 0, 0
        for lang in lang_order:
            f1, em = data[lang]
            row += f" & {f1:.1f}/{em:.1f}"
            f1_sum += f1
            em_sum += em
        
        avg_f1 = f1_sum / len(lang_order)
        avg_em = em_sum / len(lang_order)
        row += f" & {avg_f1:.1f}/{avg_em:.1f} \\\\"
        lines.append(row)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Results on MLQA question answering. We report F1/EM scores for zero-shot classification where models are fine-tuned on English SQuAD and evaluated on 7 languages. Results with $\dagger$ are from the original MLQA paper \citep{lewis2019mlqa}.}")
    lines.append(r"\label{tab:mlqa_results}")
    lines.append(r"\end{table}")
    
    latex_str = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex_str)
        print(f"LaTeX table saved to: {output_path}")
    
    return latex_str


def main():
    # Create output directory
    output_dir = Path(__file__).parent / 'plots'
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("MLQA Results Visualization")
    print("=" * 60)
    
    # Compute statistics for all models
    print("\nComputing statistics...")
    stats_dict = {}
    for model_name, model_data in RESULTS.items():
        stats = compute_statistics(model_data)
        stats_dict[model_name] = stats
        print(f"\n{model_name}:")
        print(f"  XLT:     F1={stats['xlt_f1']:.2f}  EM={stats['xlt_em']:.2f}")
        print(f"  G-XLT:   F1={stats['gxlt_f1']:.2f}  EM={stats['gxlt_em']:.2f}")
        print(f"  Overall: F1={stats['overall_f1']:.2f}  EM={stats['overall_em']:.2f}")
    
    # Generate individual model heatmaps
    print("\n" + "=" * 60)
    print("Generating individual model heatmaps...")
    print("=" * 60)
    
    for model_name, model_data in RESULTS.items():
        safe_name = model_data['short_name'].lower().replace(' ', '_').replace('-', '_')
        
        # F1 heatmap
        f1_df = matrix_to_df(model_data['f1_matrix'])
        plot_heatmap(
            f1_df,
            f"{model_name}\nG-XLT F1 Scores",
            output_dir / f'{safe_name}_f1_heatmap.png',
            metric='F1'
        )
        
        # EM heatmap
        em_df = matrix_to_df(model_data['em_matrix'])
        plot_heatmap(
            em_df,
            f"{model_name}\nG-XLT Exact Match Scores",
            output_dir / f'{safe_name}_em_heatmap.png',
            metric='EM'
        )
    
    # Generate comparison plots
    print("\n" + "=" * 60)
    print("Generating comparison plots...")
    print("=" * 60)
    
    plot_xlt_comparison(stats_dict, output_dir / 'comparison_xlt.png')
    plot_gxlt_comparison(stats_dict, output_dir / 'comparison_gxlt.png')
    plot_per_language_comparison(output_dir / 'comparison_per_language.png')
    plot_improvement_heatmap(output_dir / 'improvement_jepa_vs_xlmr.png')
    plot_summary_comparison(stats_dict, output_dir / 'comparison_summary.png')
    plot_radar_comparison(stats_dict, output_dir / 'comparison_radar.png')
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY TABLE")
    print("=" * 60)
    print(f"\n{'Model':<30} {'XLT F1':>10} {'XLT EM':>10} {'G-XLT F1':>10} {'G-XLT EM':>10}")
    print("-" * 70)
    for model_name, stats in stats_dict.items():
        short = RESULTS[model_name]['short_name']
        print(f"{short:<30} {stats['xlt_f1']:>10.2f} {stats['xlt_em']:>10.2f} {stats['gxlt_f1']:>10.2f} {stats['gxlt_em']:>10.2f}")
    print("-" * 70)
    
    # Generate paper-style tables
    print("\n" + "=" * 60)
    print("Generating paper-style tables...")
    print("=" * 60)
    
    generate_paper_table(output_dir / 'mlqa_results_table.txt')
    generate_latex_table(output_dir / 'mlqa_results_table.tex')


if __name__ == "__main__":
    main()


