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
    df = data_handler.get_reddit_data()
    print(df.columns)
    df = data_handler.get_ticker_dataframe('TSLA')
    print(df)
    # data_handler.visualize_ticker('TSLA')
    data_handler.get_ticker_dataframe('MSFT')



    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")
 
