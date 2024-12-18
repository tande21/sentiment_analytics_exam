"""
# AFTER DATAPROCESSING: 

- [x] Remove all rows that have NO ORG
- [x] Make Unique: on row. Since it will capture it multiple times
    - Computational cheap
- [x] ##Subtokens 

- [x] Then we have to call unique again. Since some rows, have
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

## MAPPPING 
    - [] Make a strict threshold

## DATAFRAME 
- NEW DATAFRAME APPEND: Processed_Entities
- Output to csv


## Look into 
        How this is being processed and used
        - US Stock tickers:  https://www.kaggle.com/datasets/marketahead/all-us-stocks-ticker 


# ANALYZE THE COMPLETED OUTPUT

## Create a HELPER.py file 


### IDEA 
- THe more positive or negative - does it increase frequency? 
"""

import warnings
warnings.simplefilter('ignore')
import time
import pandas as pd
import ast
import data_preprocessing
from semantic_search_faiss import SemanticSearch


import time
import ast
from semantic_search_faiss import SemanticSearch
from record_linking import RecordLinking

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
PROCESSED_NER_DATA = load_data('./DATA/processed_reddit_data_ALL_DATA.csv') # NU SLUTTER NER PIPELINE
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
    print("Number of keys:", len(my_dict))

def count_value_occurrences(my_dict, value_name):
    value_count = 0
    total_values = 0
    for value in my_dict.values():
        if isinstance(value, list):
            value_count += value.count(value_name)
            total_values += len(value)
        else:
            if value == value_name:
                value_count += 1
            total_values += 1
    print(f"The value '{value_name}' appears {value_count} time(s).")
    print(f"Total keys: {len(my_dict)}, Total values: {total_values}.")

def ensure_unique_values(my_dict):
    new_dict = {}
    for key, value in my_dict.items():
        if isinstance(value, list):
            seen = set()
            unique_values = []
            for item in value:
                if item == "Ticker not found":
                    if "Ticker not found" not in unique_values:
                        unique_values.append(item)
                elif item not in seen:
                    seen.add(item)
                    unique_values.append(item)
            new_dict[key] = unique_values
        else:
            new_dict[key] = value 
    return new_dict

def remove_values_with_prefix(input_dict, prefix="##"):
    cleaned_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, list):
            cleaned_values = [v for v in value if not (isinstance(v, str) and v.startswith(prefix))]
            cleaned_dict[key] = cleaned_values
        else:
            cleaned_dict[key] = value
    return cleaned_dict

def append_dict_to_dataframe(df, data_dict, new_column_name='Processed_entities'):
    df[new_column_name] = 0
    for key, value in data_dict.items():
        if key in df.index:
            df.at[key, new_column_name] = value
        else:
            df.loc[key] = [0] * len(df.columns)
            df.at[key, new_column_name] = value
    return df


if __name__ == "__main__":
    start_time = time.time()
    class_name = 'DataHandler_class'

        ####################
        ### NER PIPELINE ###
        ####################

    df = PROCESSED_NER_DATA
    df['Processed_entities'] = df['entities'].apply(ast.literal_eval)

    ##### NER DATA PROCESSED #####
    result_dict = combine_org_entities_with_ids(df, 'Processed_entities')

    print('############################')
    print('########## BEFORE ##########')
    print('############################')

    count_value_occurrences(result_dict, 'gme')
    updated_dict = ensure_unique_values(result_dict)
    updated_dict = remove_values_with_prefix(updated_dict)

    # Append 'Processed_entities' to the Reddit DataFrame
    COMPLETED_DF = append_dict_to_dataframe(REDDIT_DATA, updated_dict, 'Processed_entities')

        ####################
        ##### MAPPING ######
        ####################
    print('Mapping entities')

    # CREATE THE SETUP FOR MAPPING 
    RECORD_LINKER_CLASS = RecordLinking(COMPANIES_DF, updated_dict)
    updated_dict = RECORD_LINKER_CLASS.entity_dict_output # The new dictionary: ['GME', 'GAMESTOP'] -> ['GME', 'GME']
    updated_dict = ensure_unique_values(updated_dict)
    print_dict(updated_dict, 500)
    COMPLETED_DF = append_dict_to_dataframe(REDDIT_DATA, updated_dict, 'Processed_entities_result')


    print('###########################')
    print('########## AFTER ##########')
    print('###########################')

    count_value_occurrences(updated_dict, 'gme')

    

    ##### SAVING THE COMPLETED DATAFRAME #####
    save_dataframe_to_csv('./DATA/COMPLETED_DF.csv', COMPLETED_DF)
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")


