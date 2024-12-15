from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load the FinBERT model and tokenizer
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Option 1: Fine-tune on a financial NER dataset (you would need this step separately if using FinBERT).
# Option 2: Use a pre-trained NER model for company names and stock tickers.
ner_model_name = "dslim/bert-base-NER"  # Pre-trained general NER model
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)

# Set up the NER pipeline
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer)

# Example financial text
text = "Tesla's stock (TSLA) rose by 5% after a major announcement."

# Extract entities
entities = ner_pipeline(text)

# Filter results for companies and stock tickers
relevant_entities = [
    ent for ent in entities if ent["entity"] in ["B-ORG", "I-ORG"]  # Adjust based on model output
]

print("Extracted Entities:", relevant_entities)

