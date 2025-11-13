"""
Simple multiple choice benchmark using HuggingFace infrastructure.
Runs SWAG and saves results to CSV.
"""
import argparse
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    Trainer,
    TrainingArguments,
    DataCollatorForMultipleChoice,
    set_seed
)
import evaluate
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Benchmark multiple choice")
    parser.add_argument("--model", type=str, default="xlm-roberta-base", help="Model name")
    parser.add_argument("--dataset", type=str, default="swag", help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="regular", help="Dataset config")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output_csv", type=str, default="benchmark_results.csv", help="Output CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print(f"Running {args.dataset}/{args.dataset_config} benchmark with {args.model}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMultipleChoice.from_pretrained(args.model)
    
    # Load dataset
    dataset = load_dataset(args.dataset, args.dataset_config)
    
    # Tokenize
    ending_names = ["ending0", "ending1", "ending2", "ending3"]
    
    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples["sent1"]]
        question_headers = examples["sent2"]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names]
            for i, header in enumerate(question_headers)
        ]
        
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        
        return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    # Load metric
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    # Data collator
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
    
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
    result_df = pd.DataFrame([{
        "model": args.model,
        "task": f"{args.dataset}_{args.dataset_config}",
        "metric": "accuracy",
        "value": results["eval_accuracy"]
    }])
    
    if os.path.exists(args.output_csv):
        result_df.to_csv(args.output_csv, mode="a", header=False, index=False)
    else:
        result_df.to_csv(args.output_csv, mode="w", header=True, index=False)
    
    print(f"✓ {args.dataset}/{args.dataset_config} Accuracy: {results['eval_accuracy']:.4f}")
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()

