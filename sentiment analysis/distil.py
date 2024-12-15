import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Define the function
def distil(text):
    print(f"Processing text: {text}")
    
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")
    print(f"Tokenized input: {inputs}")
    
    # Perform inference
    with torch.no_grad():
        logits = model(**inputs).logits
        print(f"Logits: {logits}")
    
    # Get predicted class ID and corresponding label
    predicted_class_id = logits.argmax().item()
    predicted_label = model.config.id2label[predicted_class_id]
    
    # Print results
    print(f"Predicted Class ID: {predicted_class_id}")
    print(f"Predicted Label: {predicted_label}")
    print("-" * 50)  # Separator for readability

# List of sentences to process
sentences = [
    "[ROCKET][GEM STONE][PERSON RAISING BOTH HANDS IN CELEBRATION]",
    "PERSON RAISING BOTH HANDS IN CELEBRATION",
    "person raising both hands in celebration",
    "ROCKET, GEM STONE, PERSON RAISING BOTH HANDS IN CELEBRATION",
    "Setting aside a portion of your income each month for retirement can significantly improve your long-term financial security.",
    "money, sending message. [ROCKET][GEM STONE][PERSON RAISING BOTH HANDS IN CELEBRATION]",
    "Math Professor Scott Steiner says numbers spell DISASTER Gamestop shorts",
    "Exit system CEO NASDAQ pushed halt trading to give investors chance recalibrate positions...",
    "NEW SEC FILING GME! SOMEONE LESS RETARDED PLEASE INTERPRET?",
    "distract GME, thought AMC brothers aware",
    "BREAKING",
    "SHORT STOCK EXPIRATION DATE Hedgefund whales spreading disinfo...",
    "MOMENT Life fair. mother always told would complain arbitrary treatment...",
    "Currently Holding AMC NOK - retarded think move GME today?",
    "nothing say BRUH speechless MOON [ROCKET][ROCKET][ROCKET][GEM STONE][GEM STONE][WAVING HAND SIGN][WAVING HAND SIGN]",
    "need keep movement going, make history! believe right one rare opportunities help good...",
    "GME Premarket [MAPLE LEAF] Musk approved [VIDEO GAME][OCTAGONAL SIGN][GEM STONE][RAISED HAND]",
    "done GME - $AG $SLV, gentleman's short squeeze, driven macro fundamentals guys champs..."
]

# Process each sentence and display results
for i, sentence in enumerate(sentences):
    print(f"Sentence {i + 1}/{len(sentences)}:")
    distil(sentence)
