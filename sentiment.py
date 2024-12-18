from transformers import pipeline
import pandas as pd
import helper
from helper import save_dataframe_to_csv

class SentimentInfer:
    def __init__(self, model="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the sentiment inference class.
        Args:
            model (str): Name of the pre-trained model to use for sentiment analysis.
        """
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=model, truncation=True)
        self.df = pd.read_csv('COMPLETED_DF.csv')
    
    def infer(self, text):
        """
        Perform sentiment analysis on a given text.
        Args:
            text (str): Input text for sentiment analysis.
        Returns:
            result (list): List containing sentiment analysis results.
        """
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
