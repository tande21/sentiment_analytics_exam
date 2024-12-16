from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
import matplotlib as plt
from datasets import load_dataset
import evaluate
import numpy as np
import torch
import os

class FiNER_finetune():
    def __init__(self, model_checkpoint, **training_args):
        self.model_checkpoint = model_checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        self.training_args = {
            "output_dir": "finetuned_model",
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "num_train_epochs": 10,
            "weight_decay": 0.01,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",  # Use F1 score for model selection
            "lr_scheduler_type": "linear",  # Linear learning rate decay
            "warmup_ratio": 0.1,  # 10% warmup steps
            "report_to": "none",
            "save_total_limit": 2,  # Keep last 2 model checkpoints
        }
        self.training_args.update(training_args)

        self.label_list = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']

        self.id2label = {
                    0: 'O',
                    1: 'B-PER',
                    2: 'I-PER',
                    3: 'B-LOC',
                    4: 'I-LOC',
                    5: 'B-ORG',
                    6: 'I-ORG'}
        self.label2id = {
                    'O': 0,
                    'B-PER': 1,
                    'I-PER': 2,
                    'B-LOC': 3,
                    'I-LOC': 4,
                    'B-ORG': 5,
                    'I-ORG': 6
                }
        self.seqeval = evaluate.load("seqeval")

    def load_data(self):
        print("Loading dataset...")
        self.dataset = load_dataset("gtfintechlab/finer-ord-bio")

    def has_no_none_tokens(self, example):
        return None not in example["tokens"]
    

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        
        #print(f"Tokenized inputs: {tokenized_inputs}")  # Debug print

        labels = []
        for i, label in enumerate(examples["tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:  # Special tokens (e.g., [CLS], [SEP])
                    label_ids.append(-100)  # Ignore special tokens
                elif word_idx != previous_word_idx:  # Start of a new word
                    label_ids.append(label[word_idx])  # Add the correct label
                else:  # Continuation of a word (sub-word token)
                    label_ids.append(-100)  # Ignore sub-word tokens

                previous_word_idx = word_idx

            #print(f"Labels: {label_ids}")  # Debug print
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs


    def process_data(self, max_length=512):
        print("Processing dataset...")
        # Filter out examples with None tokens
        self.dataset = self.dataset.filter(self.has_no_none_tokens)
        
        # Map tokenization with additional parameters
        self.dataset = self.dataset.map(
            lambda examples: self.tokenize_and_align_labels(examples), 
            batched=True, 
            batch_size=16,
            remove_columns=self.dataset["train"].column_names  # Remove original columns
        )
    
        # Optional: Additional data augmentation or preprocessing
        # For example, truncate or pad sequences
        self.dataset = self.dataset.with_format("torch")
    
    def compute_metrics(self, predictions_and_labels):
        predictions, labels = predictions_and_labels
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    def setup_model(self):
        print("Setting up the model...")
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True  # Add this parameter
        )

    def train(self):
        print("Starting training...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        print(f"Training on: {device}")

        # Print out more detailed model information
        print(f"Model configuration:")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters())}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        args = TrainingArguments(
            **self.training_args,
            logging_dir=os.path.join(self.training_args["output_dir"], "logs"),  # Add logging directory
            logging_strategy="epoch",
            logging_steps=100,  # Log every 10 steps
            push_to_hub=False  # Disable if you're not using Hugging Face Hub
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Add early stopping callback
        )


        # Optional: Print some dataset statistics
        print(f"Training dataset size: {len(self.dataset['train'])}")
        print(f"Validation dataset size: {len(self.dataset['validation'])}")

        # Train and capture results
        train_result = trainer.train()

        # Save training results and metrics
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

        # Define the path to the subfolder where you want to save the model and tokenizer
        model_save_path = os.path.join(self.training_args["output_dir"], "model")

        # Save the model to the subfolder
        trainer.save_model(model_save_path)

        # Save the tokenizer to the same subfolder
        self.tokenizer.save_pretrained(model_save_path)
        print(f"Training complete. Model and tokenizer saved to {model_save_path}")

        # Plot training metrics
        logs = trainer.state.log_history
        epochs = [log["epoch"] for log in logs if "epoch" in log]
        train_losses = [log["loss"] for log in logs if "loss" in log]
        eval_losses = [log["eval_loss"] for log in logs if "eval_loss" in log]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label="Train Loss", marker="o")
        plt.plot(epochs, eval_losses, label="Eval Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.training_args["output_dir"], "learning_curve.png"))
        print(f"Learning curve saved to {os.path.join(self.training_args['output_dir'], 'learning_curve.png')}")

        # Evaluate on the test set
        print("Evaluating the best model on the test set...")
        test_results = trainer.evaluate(eval_dataset=self.dataset["test"])
        trainer.log_metrics("test", test_results)
        trainer.save_metrics("test", test_results)
        print(f"Test results: {test_results}")