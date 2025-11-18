"""
Simple NER benchmark using HuggingFace infrastructure.
Runs CoNLL-2003 NER and saves results to CSV.
"""
import argparse
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    set_seed
)
import evaluate
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Benchmark NER")
    parser.add_argument("--model", type=str, default="xlm-roberta-base", help="Model name")
    parser.add_argument("--dataset", type=str, default="conll2003", help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset config (optional)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output_csv", type=str, default="benchmark_results.csv", help="Output CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print(f"Running {args.dataset} benchmark with {args.model}")
    
    # Load dataset
    if args.dataset_config:
        dataset = load_dataset(args.dataset, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset)
    
    # Get label names
    label_names = dataset["train"].features["ner_tags"].feature.names
    num_labels = len(label_names)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=num_labels)
    
    # Tokenize and align labels
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            padding="max_length",
            max_length=128,
            is_split_into_words=True
        )
        
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Load metric
    metric = evaluate.load("seqeval")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Training arguments
    output_name = f"{args.dataset}_{args.dataset_config}" if args.dataset_config else args.dataset
    training_args = TrainingArguments(
        output_dir=f"./results/{output_name}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=args.seed,
        report_to="none",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("Training...")
    trainer.train()
    
    # Evaluate
    print("Evaluating...")
    results = trainer.evaluate()
    
    # Save to CSV
    task_name = f"{args.dataset}_{args.dataset_config}" if args.dataset_config else args.dataset
    result_df = pd.DataFrame([{
        "model": args.model,
        "task": task_name,
        "metric": "f1",
        "value": results["eval_f1"]
    }])
    
    # Append to CSV
    if os.path.exists(args.output_csv):
        result_df.to_csv(args.output_csv, mode="a", header=False, index=False)
    else:
        result_df.to_csv(args.output_csv, mode="w", header=True, index=False)
    
    print(f"✓ {args.dataset} F1: {results['eval_f1']:.4f}")
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()

