import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_name = "distilbertfinNER\model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

