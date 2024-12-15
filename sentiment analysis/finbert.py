from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch

def sentAnalysis(text):
    # Load tokenizer and model
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

    # Step 1: Tokenize the input text
    # inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = tokenizer(text, return_tensors="pt")

    # Step 2: Get model predictions
    outputs = model(**inputs)

    # Step 3: Apply softmax to get probabilities
    probs = softmax(outputs.logits, dim=1)

    # Step 4: Interpret results
    labels = ["negative", "neutral", "positive"]
    predicted_class = torch.argmax(probs, dim=1).item()
    sentiment = labels[predicted_class]

    print(f"Sentiment: {sentiment}")
    print(f"Probabilities: {probs.tolist()}")

sentAnalysis("Had a bad day")
sentAnalysis("The company's profits increased significantly last quarter.")
sentAnalysis("Whatever, let's see")
sentAnalysis("Haven't had a worse day")

print("------------------------------------")
sentAnalysis("ROCKET, GEM STONE, PERSON RAISING BOTH HANDS IN CELEBRATION")
sentAnalysis("The stock market surged today, with the S&P 500 reaching an all-time high due to strong corporate earnings.")
sentAnalysis("Investors are advised to diversify their portfolios to minimize risk in volatile market conditions.")
sentAnalysis("Tesla reported a 30% increase in revenue for the third quarter, driven by strong demand for electric vehicles.")
sentAnalysis("The Federal Reserve announced a 0.25% interest rate hike to combat rising inflation, signaling a tightening of monetary policy.")
sentAnalysis("Setting aside a portion of your income each month for retirement can significantly improve your long-term financial security.")