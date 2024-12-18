"""
file: main.py
purpose: To execute the entire pipeline
"""
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
import time
import pandas as pd
import ast
import data_preprocessing
from semantic_search_faiss import SemanticSearch
from collections import Counter
import time
import ast
from semantic_search_faiss import SemanticSearch
from record_linking import RecordLinking
import sentiment
from sentiment import SentimentInfer
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
START = '2020-09-29'
END = '2021-08-16'
CLASS_NAME = 'DataHandler' 
COMPANIES_TICKER_FILE = "./DATA/companies.csv" 
COMPANIES_DF = pd.read_csv(COMPANIES_TICKER_FILE)
create_and_save_data_class(CLASS_NAME)
DATAHANDLER = data_preprocessing.loadDataHandler(CLASS_NAME)
GME_TIMESERIES = DATAHANDLER.get_ticker_dataframe('GME')
# REDDIT_DATA = DATAHANDLER.reddit_data
PROCESSED_NER_DATA = load_data('./DATA/processed_reddit_data_ALL_DATA.csv') # NU SLUTTER NER PIPELINE
COMPLETED_DF = None
# SENTIMENT_CLASSIFIER = SentimentInfer() ### Out commented so we do not execute every time 
# COMPLETED_DF_LOADED = load_data('./DATA/COMPLETED_DF.csv')
COMPLETED_DF_SENTIMENT = load_data('./DATA/COMPLETED_DF_SENTIMENT.csv')

##### PROCESSING (ANALYZING) AFTER SENTIMENT #####
def extract_label(sentiment_str):
    try:
        sentiment_list = ast.literal_eval(sentiment_str)  # Safely parse the string to a list of dictionaries
        if isinstance(sentiment_list, list) and 'label' in sentiment_list[0]:
            return sentiment_list[0]['label']
    except (ValueError, SyntaxError):
        return None  # Handle bad formatting
    return None


def print_sentiment_counts(dataframe):
    """
    Prints the count of POSITIVE and NEGATIVE labels in the 'Sentiment' column
    and the total number of labels.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame with a 'Sentiment' column.
    """
    positive_count = (dataframe['Sentiment'] == 'POSITIVE').sum()
    negative_count = (dataframe['Sentiment'] == 'NEGATIVE').sum()
    total_count = len(dataframe['Sentiment'])
    
    print(f"POSITIVE labels: {positive_count}")
    print(f"NEGATIVE labels: {negative_count}")
    print(f"Total labels: {total_count}")


def find_top_10_values(dictionary):
    # Flatten all values into a single list
    all_values = []
    for key, values in dictionary.items():
        if isinstance(values, (list, set, tuple)):  # Ensure values are iterable
            all_values.extend(values)
        else:  # If the value is not iterable, add it directly
            all_values.append(values)

    # Count the occurrences of each value
    value_counts = Counter(all_values)
    
    # Get the 10 most common values
    top_10 = value_counts.most_common(10)
    print(top_10)
    return top_10


def extract_entities_to_dict(dataframe, column_name):
    entities_dict = dataframe[column_name].dropna().to_dict()
    return entities_dict

def find_rows_with_ticker(df, dictionary, ticker):
    ids_with_ticker = [key for key, value in dictionary.items() if ticker in value]
    filtered_df = df[df.index.isin(ids_with_ticker)]
    return filtered_df

def assign_color(sentiments):
    positive_count = sentiments.count('POSITIVE')
    negative_count = sentiments.count('NEGATIVE')
    if positive_count > negative_count:
        return 'green'
    elif negative_count > positive_count:
        return 'red'
    else:
        return 'grey'  # tie -> grey


if __name__ == "__main__":
    """
    Pipeline is out commented to make it easier to execute.
    """
    start_time = time.time()

        ####################
        ### NER PIPELINE ###
        ####################

    df = PROCESSED_NER_DATA
    df['Processed_entities'] = df['entities'].apply(ast.literal_eval)

    # ##### NER DATA PROCESSED #####
    result_dict = combine_org_entities_with_ids(df, 'Processed_entities')

    # print('############################')
    # print('########## BEFORE ##########')
    # print('############################')

    # count_value_occurrences(result_dict, 'gme')
    updated_dict = ensure_unique_values(result_dict)
    updated_dict = remove_values_with_prefix(updated_dict)

    # Append 'Processed_entities' to the Reddit DataFrame
    # COMPLETED_DF = append_dict_to_dataframe(REDDIT_DATA, updated_dict, 'Processed_entities')

        ####################
        ##### MAPPING ######
        ####################
    ##### LOAD DF - Instead of running model #####
    ##############################################
    # print('Mapping entities')
    ####### CREATE THE SETUP FOR MAPPING #######
    # RECORD_LINKER_CLASS = RecordLinking(COMPANIES_DF, updated_dict)
    # updated_dict = RECORD_LINKER_CLASS.entity_dict_output # The new dictionary: ['GME', 'GAMESTOP'] -> ['GME', 'GME']
    # updated_dict = ensure_unique_values(updated_dict)
    # print_dict(updated_dict, 500)
    # COMPLETED_DF = append_dict_to_dataframe(REDDIT_DATA, updated_dict, 'Processed_entities_result')

        ##########################
        ##### SENTIMENT PIPE #####
        ##########################

    ##### LOAD DF - Instead of running model #####
    # COMPLETED_DF = classify_dataframe(COMPLETED_DF)
    ################################################

    # Extract valid IDs from the updated dictionary
    valid_ids = updated_dict.keys()
    
    # Process the DataFrame
    COMPLETED_DF_SENTIMENT['Sentiment'] = COMPLETED_DF_SENTIMENT['Sentiment'].apply(extract_label)
    
    # Filter the DataFrame to keep only rows where the index (or 'id' column) is in the valid_ids
    COMPLETED_DF_SENTIMENT = COMPLETED_DF_SENTIMENT[
        COMPLETED_DF_SENTIMENT.index.isin(valid_ids)
    ]


    updated_dict = extract_entities_to_dict(COMPLETED_DF_SENTIMENT, 'Processed_entities_result')

    ###### ANALYSIS #####
    print_sentiment_counts(COMPLETED_DF_SENTIMENT)
    find_top_10_values(updated_dict)


    ##### MAP STOCKS ON TIME SERIES ###########
    ticker_to_search = 'GME'
    # df_gme = find_rows_with_ticker(ticker_to_search, updated_dict, ticker_to_search)


    ###### GET THE ROWS WHERE GME APPEARS #####
    ### Should have been used to a database for all tickers
    # df_gme = find_rows_with_ticker(COMPLETED_DF_SENTIMENT, updated_dict, ticker_to_search)


    ##### SAVING THE COMPLETED DATAFRAME #####
    save_dataframe_to_csv('./DATA/COMPLETED_DF_SENTIMENT_PROCESSED.csv', COMPLETED_DF_SENTIMENT)
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")


