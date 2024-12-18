"""
file: sentiment.py
purpose: Building a classifier, that can make,
    sentiment classifications. This is used,
    to find the sentiment of the different post. 
"""
from transformers import pipeline
import pandas as pd
import helper
from helper import save_dataframe_to_csv

class SentimentInfer:
    def __init__(self, model="distilbert-base-uncased-finetuned-sst-2-english"):
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=model, truncation=True)
        self.df = pd.read_csv('COMPLETED_DF.csv')
    
    def infer(self, text):
        result = self.sentiment_pipeline(text)
        return result

if __name__ == "__main__":
    sentiment_infer = SentimentInfer()
    df = sentiment_infer.df

    df["Sentiment"] = df["combined_text_stop"].apply(
        lambda x: sentiment_infer.infer(str(x)) if pd.notnull(x) else None
    )

    print(df["Sentiment"].head())
    save_dataframe_to_csv("complete_df_sentiment.csv", df)
