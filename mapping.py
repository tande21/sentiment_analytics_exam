"""
file: mapping.py
purpose: This file will provide functions,
    that handle the output from the NER model.
"""

import yfinance as yf
import kagglehub
import pandas as pd
from rapidfuzz import process, fuzz
import requests


def download_us_stocks_data():
    path = kagglehub.dataset_download("marketahead/all-us-stocks-tickers-company-info-logos")
    print("Path to dataset files:", path)

import pandas as pd
from rapidfuzz import process, fuzz
import requests
import pandas as pd
from rapidfuzz import process, fuzz
import requests

# Load the companies.csv file
companies_file = "./DATA/companies.csv"  # Replace with your actual file path
companies_df = pd.read_csv(companies_file)

def map_entity_to_ticker(entity, companies_df):
    # Normalize input to lowercase
    entity = entity.lower()

    # Step 1: Direct match with ticker
    if entity.upper() in companies_df["ticker"].values:
        return entity.upper()

    # Step 2: Exact match with company or short name
    exact_match = companies_df[
        (companies_df["company name"].str.lower() == entity)
        | (companies_df["short name"].str.lower() == entity)
    ]
    if not exact_match.empty:
        return exact_match.iloc[0]["ticker"]

    # Step 3: Fuzzy match with combined search space (tickers + names)
    search_space = pd.concat([
        companies_df["ticker"],
        companies_df["company name"],
        companies_df["short name"]
    ]).dropna().str.lower().unique()

    # Perform fuzzy matching
    matches = process.extract(entity, search_space, scorer=fuzz.ratio)
    best_match, similarity_score, _ = matches[0]

    # Debugging: Log fuzzy matches
    print(f"Fuzzy matches for '{entity}': {matches}")

    if similarity_score > 80:  # Adjust threshold for typos
        # Find the corresponding ticker in the dataset
        match_row = companies_df[
            (companies_df["ticker"].str.lower() == best_match)
            | (companies_df["company name"].str.lower() == best_match)
            | (companies_df["short name"].str.lower() == best_match)
        ]
        if not match_row.empty:
            return match_row.iloc[0]["ticker"]

    return "Ticker not found"

# Example: Simulated NER output (entities extracted from text)
ner_output = ["FB", "META", "Facebook", "Meta", "Apple", "AAPL", "Agilent", "GME", "GAMESTOP", "GAMSTOP"]

# Map all NER entities to tickers
print("Mapping NER entities to tickers:")
for entity in ner_output:
    ticker = map_entity_to_ticker(entity, companies_df)
    print(f"Entity: {entity} -> Ticker: {ticker}")

