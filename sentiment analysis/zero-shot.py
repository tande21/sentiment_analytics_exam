from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Load pre-trained model and tokenizer from HuggingFace
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Zero-shot classifier using cosine distance
def zero_shot_classifier(sentence, labels):
    sentence_embedding = get_embeddings(sentence)
    label_embeddings = [get_embeddings(label) for label in labels]
    similarities = [cosine_similarity(sentence_embedding, label_embedding) for label_embedding in label_embeddings]
    return labels[np.argmax(similarities)]

if __name__ == "__main__":
    # Sentences to test
    sentences = [
        "Had a bad day",
        "The company's profits increased significantly last quarter.",
        "Whatever, let's see",
        "Haven't had a worse day",
        "The stock market surged today, with the S&P 500 reaching an all-time high due to strong corporate earnings.",
        "Tesla reported a 30% increase in revenue for the third quarter, driven by strong demand for electric vehicles.",
        "Investors are advised to diversify their portfolios to minimize risk in volatile market conditions.",
        "The Federal Reserve announced a 0.25% interest rate hike to combat rising inflation, signaling a tightening of monetary policy.",
        "Setting aside a portion of your income each month for retirement can significantly improve your long-term financial security."

    ]

    # Example usage of zero-shot classifier
    labels = ["negative", "neutral", "positive"]
    for i, sentence in enumerate(sentences):
        predicted_label = zero_shot_classifier(sentence, labels)
        print(f"Predicted label for '{sentence}': {predicted_label}")
