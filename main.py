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

# Our file
import helper
from helper import (
    print_dict, 
    load_data, 
    count_value_occurrences, 
    append_dict_to_dataframe, 
    save_dataframe_to_csv
)
import ner_output_processing 
from ner_output_processing import (
    filter_org_entities,
    combine_org_entities_with_ids,
    ensure_unique_values,
    remove_values_with_prefix
)


def create_and_save_data_class(class_name):
    DataHandler = data_preprocessing.DataHandler(class_name)
    DataHandler.saveDataHandlerClass(class_name)


# CONSTANTS
CLASS_NAME = 'DataHandler' 
COMPANIES_TICKER_FILE = "./DATA/companies.csv" 
COMPANIES_DF = pd.read_csv(COMPANIES_TICKER_FILE)
create_and_save_data_class(CLASS_NAME)
DATAHANDLER = data_preprocessing.loadDataHandler(CLASS_NAME)
REDDIT_DATA = DATAHANDLER.reddit_data
PROCESSED_NER_DATA = load_data('./DATA/processed_reddit_data_ALL_DATA.csv') # NU SLUTTER NER PIPELINE
COMPLETED_DF = None


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

        ####################
        ##### MAPPING ######
        ####################






    print('###########################')
    print('########## AFTER ##########')
    print('###########################')

    count_value_occurrences(updated_dict, 'gme')

    

    ##### SAVING THE COMPLETED DATAFRAME #####
    save_dataframe_to_csv('./DATA/COMPLETED_DF.csv', COMPLETED_DF)
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")


