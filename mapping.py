import pandas as pd
from fuzzywuzzy import process, fuzz
import requests
from itertools import islice

# Load the companies.csv file
companies_file = "./DATA/companies.csv"  # Replace with your actual file path
companies_df = pd.read_csv(companies_file)

def map_entity_to_ticker(entity, companies_df):
    """
    Maps a single entity (company name or ticker) to its ticker.
    """
    # If entity is a list, process each element recursively
    if isinstance(entity, list):
        return [map_entity_to_ticker(e, companies_df) for e in entity]

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
    ]).dropna().unique()

    # Perform fuzzy matching
    matches = process.extract(entity, search_space, scorer=fuzz.ratio)
    best_match, similarity_score = matches[0]

    # Debugging: Log fuzzy matches
    # print(f"Fuzzy matches for '{entity}': {matches}")

    if similarity_score > 80:  # Adjust threshold as needed
        # Find the corresponding ticker in the dataset
        match_row = companies_df[
            (companies_df["ticker"] == best_match)
            | (companies_df["company name"] == best_match)
            | (companies_df["short name"] == best_match)
        ]
        if not match_row.empty:
            return match_row.iloc[0]["ticker"]

    return "Ticker not found"


def map_entities_dict_to_tickers_inplace_500(entities_dict, companies_df):
    """
    Updates the dictionary in place where each value is a company name, ticker, or list of names/tickers.
    Changes the value to the mapped ticker(s). Only processes the first 500 keys.
    """
    # Process only the first 500 keys
    for key in islice(entities_dict.keys(), 500):
        value = entities_dict[key]
        if isinstance(value, list):  # If the value is a list, map each element
            entities_dict[key] = [map_entity_to_ticker(v, companies_df) for v in value]
        else:  # Single value
            entities_dict[key] = map_entity_to_ticker(value, companies_df)
    return entities_dict

def map_entities_dict_to_tickers_inplace(entities_dict, companies_df):
    """
    Updates the dictionary in place where each value is a company name, ticker, or list of names/tickers.
    Changes the value to the mapped ticker(s).
    Processes the entire dictionary.
    """
    for key, value in entities_dict.items():  # Loop over all keys in the dictionary
        if isinstance(value, list):  # If the value is a list, map each element
            entities_dict[key] = [map_entity_to_ticker(v, companies_df) for v in value]
        else:  # Single value
            entities_dict[key] = map_entity_to_ticker(value, companies_df)
    return entities_dict


if __name__ == "__main__":
    # Example dictionary input
    entities_dict = {
        "company1": "FB",
        "company2": ["Facebook", "META"],
        "company3": "Apple",
        "company4": ["AAPL", "Apple Inc."],
        "company5": "Agilent",
        "company6": ["GME", "GAMESTOP"],
        "company7": "GAMSTOP"
    }
    
    # Update the dictionary in place
    updated_entities_dict = map_entities_dict_to_tickers_inplace(entities_dict, companies_df)
    
    print("Updated dictionary with tickers:")
    print(updated_entities_dict)

