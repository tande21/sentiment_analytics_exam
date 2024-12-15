import transformers
from transformers import AutoTokenizer

import datasets
from datasets import load_dataset

from transformers import DataCollatorForTokenClassification

import evaluate
import numpy as np
seqeval = evaluate.load("seqeval")

import torch

# Loading Model
model_checkpoint ="distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Loading data
ds = load_dataset("gtfintechlab/finer-ord-bio")

labels_dict = {
    'O': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-LOC': 3,
    'I-LOC': 4,
    'B-ORG': 5,
    'I-ORG': 6
}
label_list = list(labels_dict.keys())

# Tokenizing data
# \cite(https://colab.research.google.com/drive/1o0jjpWMgG1cX7eAYsV7hf2ptrOeS1fqS?usp=sharing#scrollTo=jWYE39167OAC)

def tokenize_and_align_labels(examples):
  """
  This should work?????
  """
  tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

  labels=[]
  for i, label in enumerate(examples[f"tags"]):
      word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
      previous_word_idx = None
      label_ids = []
      for word_idx in word_ids:  # Set the special tokens to -100.
          if word_idx is None:
              label_ids.append(-100)
          elif word_idx != previous_word_idx:  # Only label the first token of a given word.
              label_ids.append(label[word_idx])
          else:
              label_ids.append(-100)
          previous_word_idx = word_idx
      labels.append(label_ids)

  tokenized_inputs["labels"] = labels
  return tokenized_inputs

def tokenize_and_align_labels_safe(examples):
    try:
        # Validate input tokens
        if not validate_inputs(examples):
            raise ValueError("Invalid token structure detected.")
        return tokenize_and_align_labels(examples)
    except Exception as e:
        print(f"Skipping problematic example or batch: {e}")
        # Return dummy placeholders
        return {"input_ids": [[]], "attention_mask": [[]], "labels": [[]]}

def validate_inputs(examples):
    """
    Removes entries with invalid tokens.
    """
    for tokens in examples["tokens"]:
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            print(f"Invalid tokens: {tokens}")
            return False
    return True

def has_no_none_tokens(example):
    return None not in example["tokens"]

ds_filtered = ds.filter(has_no_none_tokens)

tokenized_data = ds_filtered.map(tokenize_and_align_labels, batched=True, batch_size=1)

# Evaluation
# \cite(https://colab.research.google.com/drive/1o0jjpWMgG1cX7eAYsV7hf2ptrOeS1fqS?usp=sharing#scrollTo=Ch-MVIk67p47)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

id2label = {
    0: 'O',
    1: 'B-PER',
    2: 'I-PER',
    3: 'B-LOC',
    4: 'I-LOC',
    5: 'B-ORG',
    6: 'I-ORG'
}

label2id = {
    'O': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-LOC': 3,
    'I-LOC': 4,
    'B-ORG': 5,
    'I-ORG': 6
}

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)

# Check if a GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"You are training on your {device}")

model.to(device)  # Move the model to GPU if available

training_args = TrainingArguments(
    output_dir="finNERbert",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    save_total_limit=2,
    # Ensure Trainer uses GPU
    no_cuda=False,  # If you want to explicitly disable GPU, set this to True
)

trainer = Trainer(
    model=model.to(device),  # Ensure the model is on the correct device
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

#trainer.train()
#trainer.evaluate()

#trainer.save_model("finNERbert\model")

example = tokenized_data["validation"][15]

for token, cls in zip(example["tokens"], example["tags"]):
  print(token, cls, id2label[cls])