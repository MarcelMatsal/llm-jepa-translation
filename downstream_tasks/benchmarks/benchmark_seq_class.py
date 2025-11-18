"""
Simple sequence classification benchmark using HuggingFace infrastructure.
Runs SST-2 sentiment analysis and saves results to CSV.
"""
import argparse
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)
import evaluate
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Benchmark sequence classification")
    parser.add_argument("--model", type=str, default="xlm-roberta-base", help="Model name")
    parser.add_argument("--dataset", type=str, default="glue", help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="sst2", help="Dataset config")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output_csv", type=str, default="benchmark_results.csv", help="Output CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print(f"Running {args.dataset}/{args.dataset_config} benchmark with {args.model}")
    
    # Load dataset
    dataset = load_dataset(args.dataset, args.dataset_config)
    
    # Determine number of labels from dataset
    label_list = dataset["train"].unique("label")
    num_labels = len(label_list)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["sentence", "idx"])
    
    # Load metric
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{args.dataset}_{args.dataset_config}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
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
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("Training...")
    trainer.train()
    
    # Evaluate
    print("Evaluating...")
    results = trainer.evaluate()
    
    # Save to CSV
    result_df = pd.DataFrame([{
        "model": args.model,
        "task": f"{args.dataset}_{args.dataset_config}",
        "metric": "accuracy",
        "value": results["eval_accuracy"]
    }])
    
    # Append to CSV (create with header if doesn't exist)
    if os.path.exists(args.output_csv):
        result_df.to_csv(args.output_csv, mode="a", header=False, index=False)
    else:
        result_df.to_csv(args.output_csv, mode="w", header=True, index=False)
    
    print(f"✓ {args.dataset}/{args.dataset_config} Accuracy: {results['eval_accuracy']:.4f}")
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()

