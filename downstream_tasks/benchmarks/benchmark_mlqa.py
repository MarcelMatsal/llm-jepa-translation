"""
MLQA Cross-lingual Question Answering Benchmark.

Implements the evaluation methodology from:
"MLQA: Evaluating Cross-lingual Extractive Question Answering" (Lewis et al., 2019)
https://arxiv.org/abs/1910.07475

Methodology:
1. Train on English SQuAD v1.1
2. Evaluate zero-shot on MLQA test sets across 7 languages
3. Report F1 and Exact Match (EM) scores

Supports both standard HuggingFace models and custom models on HuggingFace Hub.
"""
import argparse
import os
import collections
import string
import re
import json
import zipfile
import urllib.request
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

# MLQA languages
MLQA_LANGUAGES = ['en', 'ar', 'de', 'es', 'hi', 'vi', 'zh']

# MLQA data URL
MLQA_URL = "https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip"
MLQA_CACHE_DIR = Path.home() / ".cache" / "mlqa"


def download_mlqa_data():
    """Download and extract MLQA data if not already cached."""
    MLQA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    mlqa_dir = MLQA_CACHE_DIR / "MLQA_V1"
    if mlqa_dir.exists():
        print(f"MLQA data already cached at {mlqa_dir}")
        return mlqa_dir
    
    zip_path = MLQA_CACHE_DIR / "MLQA_V1.zip"
    
    if not zip_path.exists():
        print(f"Downloading MLQA data from {MLQA_URL}...")
        urllib.request.urlretrieve(MLQA_URL, zip_path)
        print("Download complete.")
    
    print("Extracting MLQA data...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(MLQA_CACHE_DIR)
    print(f"MLQA data extracted to {mlqa_dir}")
    
    return mlqa_dir


def load_mlqa_test_data(mlqa_dir, context_lang, question_lang):
    """
    Load MLQA test data for a specific language pair.
    
    Args:
        mlqa_dir: Path to MLQA_V1 directory
        context_lang: Language of the context
        question_lang: Language of the question
    
    Returns:
        List of examples with context, question, answers, and id
    """
    # MLQA test file naming convention
    test_file = mlqa_dir / "test" / f"test-context-{context_lang}-question-{question_lang}.json"
    
    if not test_file.exists():
        raise FileNotFoundError(f"MLQA test file not found: {test_file}")
    
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                example = {
                    'id': qa['id'],
                    'context': context,
                    'question': qa['question'],
                    'answers': {
                        'text': [a['text'] for a in qa['answers']],
                        'answer_start': [a['answer_start'] for a in qa['answers']]
                    }
                }
                examples.append(example)
    
    return examples


def normalize_answer(s, lang='en'):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    Adapted from official MLQA evaluation script.
    """
    def remove_articles(text):
        # Only remove English articles for English
        if lang == 'en':
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        return text

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth, lang='en'):
    """Compute F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction, lang).split()
    ground_truth_tokens = normalize_answer(ground_truth, lang).split()
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return int(prediction_tokens == ground_truth_tokens)
    
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def exact_match_score(prediction, ground_truth, lang='en'):
    """Compute Exact Match score."""
    return normalize_answer(prediction, lang) == normalize_answer(ground_truth, lang)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, lang='en'):
    """Take max score over all ground truth answers."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, lang)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def prepare_train_features(examples, tokenizer, max_length=384, doc_stride=128):
    """Prepare training features for SQuAD."""
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        
        sequence_ids = tokenized_examples.sequence_ids(i)
        
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
    
    return tokenized_examples


def prepare_validation_features(examples, tokenizer, max_length=384, doc_stride=128):
    """Prepare validation features with offset mapping for answer extraction."""
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []
    
    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1
        
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]
    
    return tokenized_examples


def postprocess_qa_predictions(
    examples,
    features,
    raw_predictions,
    tokenizer,
    n_best_size=20,
    max_answer_length=30
):
    """
    Post-process QA predictions to extract answer strings.
    """
    all_start_logits, all_end_logits = raw_predictions
    
    # Build mapping from example id to example index
    example_id_to_index = {ex['id']: i for i, ex in enumerate(examples)}
    
    # Map features to examples
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    
    predictions = {}
    
    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        
        min_null_score = None
        valid_answers = []
        
        context = example["context"]
        
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            
            # Get CLS token score as null answer score
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score > feature_null_score:
                min_null_score = feature_null_score
            
            # Get n_best start and end indices
            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip invalid indices
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    
                    valid_answers.append({
                        "score": start_logits[start_index] + end_logits[end_index],
                        "text": context[start_char:end_char]
                    })
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
        
        predictions[example["id"]] = best_answer["text"]
    
    return predictions


def evaluate_mlqa(model, tokenizer, mlqa_dir, context_lang, question_lang, device, 
                  max_length=384, doc_stride=128, batch_size=32):
    """
    Evaluate model on a specific MLQA language configuration.
    """
    # Load MLQA test data
    examples = load_mlqa_test_data(mlqa_dir, context_lang, question_lang)
    
    # Convert to Dataset format for easier processing
    test_dataset = Dataset.from_list(examples)
    
    # Prepare features
    test_features = test_dataset.map(
        lambda x: prepare_validation_features(x, tokenizer, max_length, doc_stride),
        batched=True,
        remove_columns=test_dataset.column_names,
    )
    
    # Get predictions
    model.eval()
    all_start_logits = []
    all_end_logits = []
    
    # Process in batches
    for i in tqdm(range(0, len(test_features), batch_size), 
                  desc=f"Evaluating {context_lang}-{question_lang}"):
        batch_features = test_features.select(range(i, min(i + batch_size, len(test_features))))
        
        input_ids = torch.tensor(batch_features["input_ids"]).to(device)
        attention_mask = torch.tensor(batch_features["attention_mask"]).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        all_start_logits.extend(outputs.start_logits.cpu().numpy())
        all_end_logits.extend(outputs.end_logits.cpu().numpy())
    
    # Convert features to list of dicts for post-processing
    features_list = [test_features[i] for i in range(len(test_features))]
    
    # Post-process predictions
    predictions = postprocess_qa_predictions(
        examples,
        features_list,
        (np.array(all_start_logits), np.array(all_end_logits)),
        tokenizer
    )
    
    # Calculate metrics
    # Answer language is the question language (answers are in the question's language context)
    answer_lang = question_lang
    
    f1_total = 0
    em_total = 0
    
    for example in examples:
        example_id = example["id"]
        ground_truths = example["answers"]["text"]
        prediction = predictions.get(example_id, "")
        
        f1_total += metric_max_over_ground_truths(f1_score, prediction, ground_truths, answer_lang)
        em_total += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths, answer_lang)
    
    num_examples = len(examples)
    
    return {
        "f1": 100.0 * f1_total / num_examples,
        "em": 100.0 * em_total / num_examples,
        "num_examples": num_examples
    }


def main():
    parser = argparse.ArgumentParser(description="MLQA Cross-lingual QA Benchmark")
    parser.add_argument("--model", type=str, default="xlm-roberta-base", 
                        help="HuggingFace model name or Hub repo or local path")
    parser.add_argument("--train_epochs", type=int, default=2, 
                        help="Number of training epochs on SQuAD")
    parser.add_argument("--batch_size", type=int, default=12, 
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-5, 
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=384,
                        help="Maximum sequence length")
    parser.add_argument("--doc_stride", type=int, default=128,
                        help="Document stride for long contexts")
    parser.add_argument("--languages", type=str, default="all",
                        help="Languages to evaluate: 'all' or comma-separated (e.g., 'en,de,es')")
    parser.add_argument("--cross_lingual", action="store_true",
                        help="Also evaluate cross-lingual pairs (English context only)")
    parser.add_argument("--full_matrix", action="store_true",
                        help="Evaluate full 7x7 G-XLT matrix (all 49 language pairs)")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training (use pre-trained/fine-tuned model directly)")
    parser.add_argument("--trained_model_path", type=str, default=None,
                        help="Path to already fine-tuned QA model (use with --skip_training)")
    parser.add_argument("--output_csv", type=str, default="mlqa_results.csv", 
                        help="Output CSV file")
    parser.add_argument("--output_dir", type=str, default="./results/mlqa",
                        help="Output directory for trained model")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Determine languages to evaluate
    if args.languages == "all":
        eval_languages = MLQA_LANGUAGES
    else:
        eval_languages = [l.strip() for l in args.languages.split(",")]
    
    print("=" * 60)
    print("MLQA Cross-lingual QA Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Languages: {eval_languages}")
    print(f"Cross-lingual evaluation: {args.cross_lingual}")
    print(f"Full G-XLT matrix (49 pairs): {args.full_matrix}")
    print(f"Skip training: {args.skip_training}")
    if args.trained_model_path:
        print(f"Using trained model from: {args.trained_model_path}")
    print("=" * 60)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load or train model
    if args.skip_training and args.trained_model_path:
        # Load already fine-tuned model
        print(f"\nLoading fine-tuned model from {args.trained_model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.trained_model_path)
        model = AutoModelForQuestionAnswering.from_pretrained(args.trained_model_path)
    elif args.skip_training:
        # Use base model without fine-tuning (zero-shot)
        print(f"\nLoading model {args.model} (no fine-tuning)...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    else:
        # Load model and tokenizer for training
        print("\nLoading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForQuestionAnswering.from_pretrained(args.model)
        
        # =============================================
        # PHASE 1: Train on SQuAD v1.1
        # =============================================
        print("\n" + "=" * 60)
        print("PHASE 1: Training on SQuAD v1.1")
        print("=" * 60)
        
        # Load SQuAD
        squad = load_dataset("squad")
        
        # Prepare training features
        print("Preparing training features...")
        train_dataset = squad["train"].map(
            lambda x: prepare_train_features(x, tokenizer, args.max_length, args.doc_stride),
            batched=True,
            remove_columns=squad["train"].column_names,
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.train_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            save_strategy="epoch",
            logging_steps=100,
            seed=args.seed,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )
        
        # Train
        print("Training...")
        trainer.train()
        
        # Save model
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")
    
    # Move model to device
    model = model.to(device)
    
    # =============================================
    # PHASE 2: Download and Evaluate on MLQA
    # =============================================
    print("\n" + "=" * 60)
    print("PHASE 2: Zero-shot Evaluation on MLQA")
    print("=" * 60)
    
    # Download MLQA data
    mlqa_dir = download_mlqa_data()
    
    results = []
    
    # Monolingual evaluation (same language for context and question)
    print("\n--- Monolingual Evaluation ---")
    for lang in eval_languages:
        print(f"\nEvaluating {lang}-{lang}...")
        
        try:
            metrics = evaluate_mlqa(
                model, tokenizer, mlqa_dir, lang, lang, device,
                args.max_length, args.doc_stride, args.eval_batch_size
            )
            
            print(f"  F1: {metrics['f1']:.2f}")
            print(f"  EM: {metrics['em']:.2f}")
            print(f"  Examples: {metrics['num_examples']}")
            
            results.append({
                "model": args.model,
                "task": f"mlqa_{lang}_{lang}",
                "context_lang": lang,
                "question_lang": lang,
                "f1": metrics['f1'],
                "em": metrics['em'],
                "num_examples": metrics['num_examples']
            })
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "model": args.model,
                "task": f"mlqa_{lang}_{lang}",
                "context_lang": lang,
                "question_lang": lang,
                "f1": -1,
                "em": -1,
                "num_examples": 0
            })
    
    # Cross-lingual evaluation (English context, other language questions)
    if args.cross_lingual and not args.full_matrix:
        print("\n--- Cross-lingual Evaluation (G-XLT, English context only) ---")
        for lang in eval_languages:
            if lang == 'en':
                continue
            
            print(f"\nEvaluating en-{lang}...")
            
            try:
                metrics = evaluate_mlqa(
                    model, tokenizer, mlqa_dir, 'en', lang, device,
                    args.max_length, args.doc_stride, args.eval_batch_size
                )
                
                print(f"  F1: {metrics['f1']:.2f}")
                print(f"  EM: {metrics['em']:.2f}")
                print(f"  Examples: {metrics['num_examples']}")
                
                results.append({
                    "model": args.model,
                    "task": f"mlqa_en_{lang}",
                    "context_lang": "en",
                    "question_lang": lang,
                    "f1": metrics['f1'],
                    "em": metrics['em'],
                    "num_examples": metrics['num_examples']
                })
            except Exception as e:
                print(f"  Error: {e}")
                results.append({
                    "model": args.model,
                    "task": f"mlqa_en_{lang}",
                    "context_lang": "en",
                    "question_lang": lang,
                    "f1": -1,
                    "em": -1,
                    "num_examples": 0
                })
    
    # Full G-XLT matrix evaluation (all 49 language pairs)
    if args.full_matrix:
        print("\n--- Full G-XLT Matrix Evaluation (49 pairs) ---")
        
        # Store results in a matrix format for nice printing
        f1_matrix = {}
        em_matrix = {}
        
        for context_lang in eval_languages:
            f1_matrix[context_lang] = {}
            em_matrix[context_lang] = {}
            
            for question_lang in eval_languages:
                # Skip if already evaluated in monolingual section
                if context_lang == question_lang:
                    # Find the monolingual result we already computed
                    mono_result = next((r for r in results if r['context_lang'] == context_lang and r['question_lang'] == question_lang), None)
                    if mono_result and mono_result['f1'] >= 0:
                        f1_matrix[context_lang][question_lang] = mono_result['f1']
                        em_matrix[context_lang][question_lang] = mono_result['em']
                    else:
                        f1_matrix[context_lang][question_lang] = -1
                        em_matrix[context_lang][question_lang] = -1
                    continue
                
                print(f"\nEvaluating {context_lang}-{question_lang}...")
                
                try:
                    metrics = evaluate_mlqa(
                        model, tokenizer, mlqa_dir, context_lang, question_lang, device,
                        args.max_length, args.doc_stride, args.eval_batch_size
                    )
                    
                    print(f"  F1: {metrics['f1']:.2f}")
                    print(f"  EM: {metrics['em']:.2f}")
                    
                    f1_matrix[context_lang][question_lang] = metrics['f1']
                    em_matrix[context_lang][question_lang] = metrics['em']
                    
                    results.append({
                        "model": args.model,
                        "task": f"mlqa_{context_lang}_{question_lang}",
                        "context_lang": context_lang,
                        "question_lang": question_lang,
                        "f1": metrics['f1'],
                        "em": metrics['em'],
                        "num_examples": metrics['num_examples']
                    })
                except Exception as e:
                    print(f"  Error: {e}")
                    f1_matrix[context_lang][question_lang] = -1
                    em_matrix[context_lang][question_lang] = -1
                    results.append({
                        "model": args.model,
                        "task": f"mlqa_{context_lang}_{question_lang}",
                        "context_lang": context_lang,
                        "question_lang": question_lang,
                        "f1": -1,
                        "em": -1,
                        "num_examples": 0
                    })
        
        # Print F1 matrix in paper format
        print("\n" + "=" * 80)
        print("G-XLT F1 MATRIX (rows=context, cols=question)")
        print("=" * 80)
        header = "c\\q\t" + "\t".join(eval_languages)
        print(header)
        print("-" * 80)
        for context_lang in eval_languages:
            row = [context_lang]
            for question_lang in eval_languages:
                val = f1_matrix[context_lang][question_lang]
                row.append(f"{val:.1f}" if val >= 0 else "ERR")
            print("\t".join(row))
        
        # Print EM matrix
        print("\n" + "=" * 80)
        print("G-XLT EM MATRIX (rows=context, cols=question)")
        print("=" * 80)
        print(header)
        print("-" * 80)
        for context_lang in eval_languages:
            row = [context_lang]
            for question_lang in eval_languages:
                val = em_matrix[context_lang][question_lang]
                row.append(f"{val:.1f}" if val >= 0 else "ERR")
            print("\t".join(row))
        
        # Calculate and print averages
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        # XLT average (diagonal)
        xlt_f1 = [f1_matrix[l][l] for l in eval_languages if f1_matrix[l][l] >= 0]
        xlt_em = [em_matrix[l][l] for l in eval_languages if em_matrix[l][l] >= 0]
        if xlt_f1:
            print(f"XLT Average (diagonal):     F1={sum(xlt_f1)/len(xlt_f1):.2f}  EM={sum(xlt_em)/len(xlt_em):.2f}")
        
        # G-XLT average (off-diagonal)
        gxlt_f1 = []
        gxlt_em = []
        for cl in eval_languages:
            for ql in eval_languages:
                if cl != ql and f1_matrix[cl][ql] >= 0:
                    gxlt_f1.append(f1_matrix[cl][ql])
                    gxlt_em.append(em_matrix[cl][ql])
        if gxlt_f1:
            print(f"G-XLT Average (off-diag):   F1={sum(gxlt_f1)/len(gxlt_f1):.2f}  EM={sum(gxlt_em)/len(gxlt_em):.2f}")
        
        # Overall average
        all_f1 = [v for cl in f1_matrix for v in f1_matrix[cl].values() if v >= 0]
        all_em = [v for cl in em_matrix for v in em_matrix[cl].values() if v >= 0]
        if all_f1:
            print(f"Overall Average (all 49):   F1={sum(all_f1)/len(all_f1):.2f}  EM={sum(all_em)/len(all_em):.2f}")
    
    # Save results to CSV
    result_df = pd.DataFrame(results)
    
    # Append or create CSV
    if os.path.exists(args.output_csv):
        result_df.to_csv(args.output_csv, mode="a", header=False, index=False)
    else:
        result_df.to_csv(args.output_csv, mode="w", header=True, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("MLQA EVALUATION COMPLETE")
    print("=" * 60)
    
    # Print results table
    print("\nResults Summary:")
    print("-" * 60)
    print(f"{'Task':<20} {'F1':>10} {'EM':>10}")
    print("-" * 60)
    
    for r in results:
        if r['f1'] >= 0:
            print(f"{r['task']:<20} {r['f1']:>10.2f} {r['em']:>10.2f}")
        else:
            print(f"{r['task']:<20} {'ERROR':>10} {'ERROR':>10}")
    
    print("-" * 60)
    
    # Calculate averages
    valid_results = [r for r in results if r['f1'] >= 0]
    if valid_results:
        avg_f1 = sum(r['f1'] for r in valid_results) / len(valid_results)
        avg_em = sum(r['em'] for r in valid_results) / len(valid_results)
        print(f"{'Average':<20} {avg_f1:>10.2f} {avg_em:>10.2f}")
    
    print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
