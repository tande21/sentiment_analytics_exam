import transformers
from transformers import AutoTokenizer

import datasets
from datasets import load_dataset

model_checkpoint ="distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

raw_data = load_dataset("gtfintechlab/finer-ord")
ds = load_dataset("gtfintechlab/finer-ord-bio")
train_data = raw_data["train"]

label_dict = {'O': 0, 'PER_B': 1, 'PER_I': 2, 'LOC_B': 3, 'LOC_I': 4, 'ORG_B': 5, 'ORG_I': 6}
labels = list(label_dict.keys())
example = ds['train'][0]
#print(example["tokens"])
tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)

word_ids = tokenized_input.word_ids()
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
for idx, token  in zip(word_ids, tokens):
  print(token, idx)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels=[]
    for i, labels in enumerate()

tokenized_data = raw_data.map(tokenize_and_align_labels, batched=True)
#NER Pipeline to label and sort by Company names and stickers(?)