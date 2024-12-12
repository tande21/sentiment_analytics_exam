"""
file name: data_preprocessing.py
purpose: This file will handle all the data

TODO: 
    - rename all the columns in stock
    - remove time from post 
    - Load the annotated data


"""

import kagglehub
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import yfinance as yf

# Our files
import visualization

class DataHandler:
    """
    DataHandler is a class to handle data preprocessing,
    for all the data in the project: 
        - reddit walstreet post 
        - timeseries for stocks

    Preconditions:
    
    Parameters:
        ...
        
    """

    def __init__(self, class_name):
        self.class_name = class_name
        self.stock_list = ['TSLA', 'MSFT']
        self.START = '2020-01-01'
        self.END = '2022-01-01'
        
        # Data
        self.reddit_data = self._load_reddit_data()
        self.timeseries_data = self._load_timeseries_data(self.stock_list) 

    def _load_reddit_data(self):
        file_path = './DATA/reddit_wsb.csv'
        df_reddit = pd.read_csv(file_path)
        df_reddit = df_reddit.drop(['score', 'url', 'created'], axis=1)
        return df_reddit


    def _load_timeseries_data(self, tickers):
        path = './DATA/TIMESERIES/'
        timeseries_data = {}
        
        for ticker in tickers:
            file_path = os.path.join(path, f"{ticker}.csv")
            
            if os.path.exists(file_path):
                # Skip the first 3 rows that are not data rows
                df = pd.read_csv(file_path, skiprows=2)
                
                # If the CSV after skipping rows still doesn't provide a 'Date' column,
                # we assume the first column is actually the date. Rename it to 'Date'.
                if 'Date' not in df.columns:
                    # Rename the first column to 'Date'
                    first_col = df.columns[0]
                    df.rename(columns={first_col: 'Date'}, inplace=True)


                df.rename(columns={df.columns[1]: 'Adj Close'}, inplace=True)
                df.rename(columns={df.columns[2]: 'Close'}, inplace=True)

                df.rename(columns={df.columns[3]: 'High'}, inplace=True)
                df.rename(columns={df.columns[4]: 'Low'}, inplace=True)
                df.rename(columns={df.columns[5]: 'Open'}, inplace=True)
                df.rename(columns={df.columns[6]: 'Volume'}, inplace=True)


    
                # Convert the 'Date' column to datetime and set it as the index
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df.dropna(subset=['Date'], inplace=True)  # Drop rows without valid Date
                df.set_index('Date', inplace=True)
    
                # Remove rows where the 'Price' column is 'Ticker'
                if 'Price' in df.columns:
                    df = df[df['Price'] != 'Ticker']
                
                timeseries_data[ticker] = df
                print(f"Loaded data for {ticker}")
            else:
                print(f"File not found for ticker: {ticker}")
        
        return timeseries_data
    
        
        # Saving the datahandler 
    def saveDataHandlerClass(self, file_name):
        folder_path = './DATA/'
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'wb') as file:
            pickle.dump(self, file)


    def visualize_ticker(self, ticker):
        # Check if the ticker's data is loaded in the dictionary
        if ticker in self.timeseries_data:
            df = self.timeseries_data[ticker]
            visualization.plot_price(df, ticker)  # Pass the DataFrame first, then the ticker
            print(f"Visualization for {ticker} completed.")
        else:
            print(f"Data for ticker {ticker} is not loaded.")

    
    """
    Getter functions
    """
    def get_class_name(self):
        return self.class_name

    def get_reddit_data(self):
        return self.reddit_data

    def get_timeseries(self, ticker):
        path = './DATA/TIMESERIES/'
        df = pd.read_csv(path + ticker + '.csv')
        return df

    def get_ticker_dataframe(self, ticker):
        # Check if the ticker exists in the dictionary
        if ticker in self.timeseries_data:
            return self.timeseries_data[ticker]  # Return the corresponding DataFrame
        else:
            print(f"Ticker '{ticker}' not found in the data.")
            return None

          
def loadDataHandler(class_name):
    path = './DATA/'
    class_path = path + class_name
    with open(class_path, 'rb') as input:
        data_handler = pickle.load(input)
        data_handler.get_class_name = data_handler.get_class_name()
        return data_handler



"""
Files that need to only be executed once.
"""

def download_wallstreets_bets():
    path = kagglehub.dataset_download("gpreda/reddit-wallstreetsbets-posts")
    print("Path to dataset files:", path)


from datasets import load_dataset
def download_annotate():
    ds = load_dataset("gtfintechlab/finer-ord")


def timeseries_to_csv(ticker, START, END):
    path = './DATA/TIMESERIES/'
    stock = yf.download(ticker, start=START, end=END)
    stock.to_csv(path + ticker + '.csv', index=True)

def tickers_timeseries_to_csv(tickers, START, END):
        path = './DATA/TIMESERIES/'
        
        for ticker in tickers:
            stock = yf.download(ticker, start=START, end=END)
            stock.to_csv(os.path.join(path, f"{ticker}.csv"), index=True)
            print(f"Data for {ticker} saved to {path}{ticker}.csv")



tickers = ['TSLA', 'MSFT']
START = '2020-01-01'
END = '2022-01-01'

tickers_timeseries_to_csv(tickers, START, END)


# download_annotate()
