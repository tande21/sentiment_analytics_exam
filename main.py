"""
file: main.py

"""

import warnings
warnings.simplefilter('ignore')
import time

# Our files
import data_preprocessing


def create_and_save_data_class(class_name):
    DataHandler = data_preprocessing.DataHandler(class_name)
    DataHandler.saveDataHandlerClass(class_name)

if __name__ == "__main__":
    start_time = time.time()
    class_name = 'DataHandler_class'
    create_and_save_data_class(class_name)
    data_handler = data_preprocessing.loadDataHandler(class_name)
    df_reddit = data_handler.get_reddit_data()
    df = data_handler.get_ticker_dataframe('TSLA')
    # data_handler.visualize_ticker('TSLA')
    data_handler.get_ticker_dataframe('MSFT')
   
    # data_handler.visualize_common_words(df_reddit, 'clean')
    data_handler.visualize_word_count()
    data_handler.visualize_score_count()

    df = data_handler.get_reddit_data()
    # empty_count = df['combined_text'].isna().sum() + (df['combined_text'] == '').sum()
    # print(f"Number of empty rows: {empty_count}")




    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")
 
