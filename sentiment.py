from transformers import pipeline
import pandas as pd
import helper
from helper import save_dataframe_to_csv

class SentimentInfer:
    def __init__(self, model="distilbert-base-uncased-finetuned-sst-2-english"):
        # Initialize the sentiment analysis pipeline
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=model, truncation=True)
        self.df = pd.read_csv('COMPLETED_DF.csv')
    
    def infer(self, text):
        # Use the pipeline to predict sentiment
        result = self.sentiment_pipeline(text)
        return result
    
    


# Example usage:
if __name__ == "__main__":
    sentiment_infer = SentimentInfer()
    df = sentiment_infer.df
    # text = "I love this product! It's amazing."
 #   df["Sentiment"] = df["combined_text"].apply(lambda x: sentiment_infer.infer(x))
    df["Sentiment"] = df["combined_text_stop"].apply(
    lambda x: sentiment_infer.infer(str(x)) if pd.notnull(x) else None
)
    #print(sentiment_infer.infer(text))
    print(df["Sentiment"].head())
    save_dataframe_to_csv("theis_output.csv", df)
