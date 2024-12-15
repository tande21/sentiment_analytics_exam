import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import data_preprocessing
model_name = "distilbertfinNER\model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_entities(text):
    results = ner_pipeline(text)
    entities = [{"entity": result["entity_group"], "word": result["word"]} for result in results]
    return entities

def nerData(df, column_name):
    df["entities"] = df[column_name].apply(extract_entities)