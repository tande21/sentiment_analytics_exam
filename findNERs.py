from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np

# Define the label list (same as used during training)
label_list = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']

# Load the tokenizer and model from the saved directory
model_path = "distilbertfinNER\model"  # Path to your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_entities(sentence):
    """
    Predict named entities in a given sentence while filtering out special tokens.
    
    Args:
        sentence (str): Input sentence to analyze
    
    Returns:
        list of tuples: (token, label) for non-special tokens
    """
    # Tokenize the input
    inputs = tokenizer(sentence, is_split_into_words=False, return_tensors="pt", truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move tensors to device

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    # Convert to tokens and labels
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu().numpy())
    predicted_labels = [label_list[pred] for pred in predictions[0].cpu().numpy()]

    # Filter out special tokens
    special_tokens = {'[CLS]', '[SEP]', '[PAD]'}
    results = []
    for token, label in zip(tokens, predicted_labels):
        if token not in special_tokens:
            results.append((token, label))
    
    return results

# Example usage
sentence = "Tesla's stock (TSLA) rose by 5% after a major announcement."
entities = predict_entities(sentence)

# Print results
for token, label in entities:
    print(f"{token}: {label}")

# Optional: Extract and group named entities
def group_entities(entities):
    """
    Group consecutive tokens with the same entity type.
    
    Args:
        entities (list): List of (token, label) tuples
    
    Returns:
        dict: Grouped entities by type
    """
    grouped = {
        'PER': [],
        'LOC': [],
        'ORG': []
    }
    
    current_entity = None
    current_tokens = []
    
    for token, label in entities:
        # Split label into type and position (B-/I-)
        if label == 'O':
            # Complete previous entity if exists
            if current_entity:
                grouped[current_entity].append(' '.join(current_tokens))
                current_entity = None
                current_tokens = []
            continue
        
        entity_type = label.split('-')[1]
        position = label.split('-')[0]
        
        # Start of a new entity
        if position == 'B':
            # Complete previous entity if exists
            if current_entity:
                grouped[current_entity].append(' '.join(current_tokens))
            
            # Start new entity
            current_entity = entity_type
            current_tokens = [token]
        
        # Continue of previous entity
        elif position == 'I' and current_entity == entity_type:
            current_tokens.append(token)
    
    # Add last entity if exists
    if current_entity:
        grouped[current_entity].append(' '.join(current_tokens))
    
    return {k: v for k, v in grouped.items() if v}  # Remove empty lists

# Example of grouping entities
print("\nGrouped Entities:")
print(group_entities(entities))