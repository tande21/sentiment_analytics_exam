"""
file: finNER_finetuner.py
purpose: This file is used to, 
    fine tune our NER model. 
"""
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
import matplotlib.pyplot as plt
from datasets import load_dataset
import evaluate
import numpy as np
import torch
import os

# \cite(https://colab.research.google.com/drive/1o0jjpWMgG1cX7eAYsV7hf2ptrOeS1fqS?usp=sharing#scrollTo=jWYE39167OAC)

class FiNER_finetune():
    """
    Takes a model and finetunes it to the FiNER-ORD-BIO with given training args
    """
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
            "metric_for_best_model": "f1", 
            "lr_scheduler_type": "linear", 
            "warmup_ratio": 0.1, 
            "report_to": "none",
            "save_total_limit": 2, #How many checkpoints to keep during training.
                                        #Will save at least two. The last and the best.
        }
        self.training_args.update(training_args)

        self.label_list = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG'] #List of labels from dataset following the BIO standard.
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

    #The dataset has one row that contains a None. This crashes the training
    def has_no_none_tokens(self, example):
        return None not in example["tokens"]
    
    #Based on code by Tariq Yousef
        #\cite{https://colab.research.google.com/drive/1o0jjpWMgG1cX7eAYsV7hf2ptrOeS1fqS?usp=sharing#scrollTo=Ch-MVIk67p47}
    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples["tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:  # Special tokens (e.g., [CLS], [SEP])
                    label_ids.append(-100)  # The -100 label is commonly used for token to not be used in loss calculation
                elif word_idx != previous_word_idx:  # Start of a new word
                    label_ids.append(label[word_idx])  # Add the correct label
                else:  # Continuation of a word (sub-word token)
                    label_ids.append(-100)  # Ignore sub-word tokens

                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Boilerplate AI Code
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
    
    #\cite{https://colab.research.google.com/drive/1o0jjpWMgG1cX7eAYsV7hf2ptrOeS1fqS?usp=sharing#scrollTo=Ch-MVIk67p47}
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
            ignore_mismatched_sizes=True  # Used to ignore discrepency in the size of output layer
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
            logging_dir=os.path.join(self.training_args["output_dir"], "logs"),  # Logs are saved in a folder in the output directory
            logging_strategy="epoch",
            logging_steps=20,  # Log every n steps
            push_to_hub=False  # Not using hugginface hub
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Add early stopping callback
        )


        # Dataset stats
        print(f"Training dataset size: {len(self.dataset['train'])}")
        print(f"Validation dataset size: {len(self.dataset['validation'])}")

        # Train and save results to variable
        train_result = trainer.train()

        # Save training results and metrics
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

        
        model_save_path = os.path.join(self.training_args["output_dir"], "model")
        # The finale/best fine-tuned model is saved to a model folder in the output directory
        trainer.save_model(model_save_path)

        # Save the tokenizer to the same subfolder
        self.tokenizer.save_pretrained(model_save_path)
        print(f"Training complete. Model and tokenizer saved!")


        #TODO: Fix to get at least one nice graph :|
        # Plot training metrics
        # Extract logged data
        # Extract step-wise logged data
        logs = trainer.state.log_history

        # Separate training and evaluation logs
        train_steps = [log["step"] for log in logs if "step" in log and "loss" in log]
        train_losses = [log["loss"] for log in logs if "step" in log and "loss" in log]
        eval_steps = [log["step"] for log in logs if "step" in log and "eval_loss" in log]
        eval_losses = [log["eval_loss"] for log in logs if "step" in log and "eval_loss" in log]

        # Plot step-wise loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_losses, label="Train Loss", marker=".", alpha=0.7)
        plt.plot(eval_steps, eval_losses, label="Eval Loss", marker=".", alpha=0.7)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Step-Wise Loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.training_args["output_dir"], "step_wise_loss.png"))
        print(f"Step-wise loss curve saved to {os.path.join(self.training_args['output_dir'], 'step_wise_loss.png')}")

        # Evaluate on the test set
        print("Evaluating the best model on the test set...")
        test_results = trainer.evaluate(eval_dataset=self.dataset["test"])
        trainer.log_metrics("test", test_results)
        trainer.save_metrics("test", test_results)
        print(f"Test results: {test_results}")
