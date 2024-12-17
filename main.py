"""
# AFTER DATAPROCESSING: 

- [x] Remove all rows that have NO ORG
- [x] Make Unique: on row. Since it will capture it multiple times
    - Computational cheap
- [x] ##Subtokens 

- [] Then we have to call unique again. Since some rows, have
        - for example gamestop and GME. So we will get,
        gme 2 times.
        - Men vi gør det ikke paa: ticker not found. 
            - saa vi kan se hvad den refere til. 

## ANALYZE NER OUTPUT DATA
    ### Before processing
    - Find how many rows we have before processing

    #### Step længere after fuzzy
    - Find how many unique valeus 
    
    ## 
    - Find how many unique values we have in the dict. 
    - Find how many rows contains only ['ticker not found'] 

## DATAFRAME 
- NEW DATAFRAME APPEND: Processed_Entities
- Output to csv


## Look into 
        How this is being processed and used
        - US Stock tickers:  https://www.kaggle.com/datasets/marketahead/all-us-stocks-ticker 


# ANALYZE THE COMPLETED OUTPUT

##         



### IDEA 
- THe more positive or negative - does it increase frequency? 
"""
import warnings
warnings.simplefilter('ignore')
import time
import pandas as pd
import ast
import data_preprocessing
import mapping
from mapping import map_entities_dict_to_tickers_inplace

def create_and_save_data_class(class_name):
    DataHandler = data_preprocessing.DataHandler(class_name)
    DataHandler.saveDataHandlerClass(class_name)

def load_data(path): 
    return pd.read_csv(path)


# CONSTANTS
CLASS_NAME = 'DataHandler' 
COMPANIES_TICKER_FILE = "./DATA/companies.csv" 
COMPANIES_DF = pd.read_csv(COMPANIES_TICKER_FILE)
create_and_save_data_class(CLASS_NAME)
DATAHANDLER = data_preprocessing.loadDataHandler(CLASS_NAME)
REDDIT_DATA = DATAHANDLER.reddit_data
PROCESSED_NER_DATA = load_data('./DATA/processed_reddit_data_ALL_DATA.csv')
COMPLETED_DF = None

def save_dataframe_to_csv(file_path, dataframe):
    try:
        dataframe.to_csv(file_path, index=False)  # Save without including the index column
        print(f"DataFrame successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving DataFrame to CSV: {e}")

def filter_org_entities(entity_list):
    if not isinstance(entity_list, list):
        return []
    return [entity for entity in entity_list if entity.get('entity') == 'ORG']

def combine_org_entities_with_ids(df, column_name='entities'):
    combined_dict = {}
    for index, row in df.iterrows():
        entities = row[column_name]
        # After applying literal_eval, entities should be a list of dicts
        if isinstance(entities, list):
            org_words = [entity['word'] for entity in entities 
                         if isinstance(entity, dict) and entity.get('entity') == 'ORG']
            if org_words:
                combined_dict[index] = org_words
    return combined_dict


def print_dict(my_dict, n_lines):
    count = 0
    for key, value in my_dict.items():
        print(key, value)
        count += 1
        if count == n_lines:
            break

    num_keys = len(my_dict)
    print("Number of keys:", num_keys)


def count_value_occurrences(my_dict, value_name):
    value_count = 0  # Counter for occurrences of 'value_name'
    total_values = 0  # Counter for total number of values in the dictionary

    for value in my_dict.values():
        # If value is a list, count occurrences of 'value_name' in it
        if isinstance(value, list):
            value_count += value.count(value_name)
            total_values += len(value)  # Add the length of the list to total_values
        else:
            # If value is a single value (not a list), check equality
            if value == value_name:
                value_count += 1
            total_values += 1  # Increment total values for a single value

    # Count the total number of keys
    num_keys = len(my_dict)

    print(f"The value '{value_name}' appears {value_count} time(s) in the dictionary.")
    print(f"The dictionary has {num_keys} key(s).")
    print(f"The total number of values in the dictionary is {total_values}.")



def ensure_unique_values(my_dict):
    """
    Ensures each key in the dictionary contains only unique values.
    If the value is a list, duplicates are removed, except 'Ticker not found' is always preserved.

    Parameters:
        my_dict (dict): Input dictionary.

    Returns:
        dict: A new dictionary with unique values for each key.
    """
    new_dict = {}

    for key, value in my_dict.items():
        if isinstance(value, list):
            # Remove duplicates while preserving order
            seen = set()
            unique_values = []
            
            # Iterate over values to remove duplicates, preserving 'Ticker not found'
            for item in value:
                if item == "Ticker not found":
                    if "Ticker not found" not in unique_values:
                        unique_values.append(item)
                elif item not in seen:
                    seen.add(item)
                    unique_values.append(item)
                    
            new_dict[key] = unique_values
        else:
            # If the value is not a list, leave it unchanged
            new_dict[key] = value

    return new_dict


def remove_values_with_prefix(input_dict, prefix="##"):
    """
    Removes all values starting with a specific prefix (default '##') from lists in the dictionary.

    Parameters:
        input_dict (dict): Input dictionary.
        prefix (str): Prefix to look for and remove.

    Returns:
        dict: A new dictionary with cleaned values.
    """
    cleaned_dict = {}

    for key, value in input_dict.items():
        if isinstance(value, list):
            # Filter out values that start with the given prefix
            cleaned_values = [v for v in value if not (isinstance(v, str) and v.startswith(prefix))]
            cleaned_dict[key] = cleaned_values
        else:
            # Leave non-list values unchanged
            cleaned_dict[key] = value

    return cleaned_dict

def append_dict_to_dataframe(df, data_dict, new_column_name='Processed_entities'):
    df[new_column_name] = 0

    # Update rows where keys in the dictionary match the dataframe's index
    for key, value in data_dict.items():
        if key in df.index:
            df.at[key, new_column_name] = value  # Assign the list from the dictionary
        else:
            df.loc[key] = [0] * len(df.columns)  # Fill missing rows with zeros
            df.at[key, new_column_name] = value  # Assign the value for 'processed_entities'

    return df


def summary():
    
    """
    DO THIS BEFORE PROCESSING AND AFTER 

    output: Total number of keys in DIC
    output: Total number of values in DIC
    output: Total number of 'Ticker not found'
    output: Total nomber of UNIQUES in entire Dic

    """



if __name__ == "__main__":
    start_time = time.time()
    class_name = 'DataHandler_class'


        ####################
        ### NER PIPELIN ####
        ####################

    df = PROCESSED_NER_DATA 
    # Convert string representations to actual lists of dictionaries
    df['entities'] = df['entities'].apply(ast.literal_eval)

    ##### NER DATA PROCESSED #####
    result_dict = combine_org_entities_with_ids(df, 'entities')
    # print_dict(result_dict, 500)


    ##### NOGLE AF DE HER FUNKTIONER SKAL Gøres paa dataframe aswell? VI GØR DEM KUN PAA DICT.
    ##### TVUNGET TIL AT GØRE DET PAA DENNE MØDE: DA DEN ENE KOLONNE IKKE KAN GØRES PAA ANDRE MAADER.

    print('############################')
    print('########## BEFORE ##########')
    print('############################')

    count_value_occurrences(result_dict, 'gme')
    updated_dict = ensure_unique_values(result_dict)
    # print_dict(result_dict, 10)
    updated_dict = remove_values_with_prefix(result_dict)
    # print_dict(result_dict, 10)
    COMPLETED_DF = append_dict_to_dataframe(REDDIT_DATA, updated_dict, 'Processed_entities')
    updated_dict = map_entities_dict_to_tickers_inplace(updated_dict, COMPANIES_DF)

    # ENURE unique values again. Since it might have,
    # [Gamestop, GME] -> [GME, GME]
    updated_dict = ensure_unique_values(updated_dict)
    COMPLETED_DF = append_dict_to_dataframe(REDDIT_DATA, updated_dict, 'Processed_entities_result')
    
    print('###########################')
    print('########## AFTER ##########')
    print('###########################')

    count_value_occurrences(updated_dict, 'gme')
    save_dataframe_to_csv('./DATA/COMPLETED_DF.csv',COMPLETED_DF)
