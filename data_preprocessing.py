"""
file name: data_preprocessing.py
purpose: This file will handle all the data

TODO: 
    - rename all the columns in stock
    - remove time from post 
    - Load the annotated data

    # Analysis functions
    - + Common words 2 time: Before and after
    - - Plot score histogram
    - + Plot histogram of length post
    - + 3 plot for common words before emojis.
    ? ? (lowercase i visualization?)

    - plot number of posts each day.
    - Plot sp500 index in time period

    # Processing steps
    - + Drop those columns that are not needed
    - + concattenate title and body
    - + remove http www, http com
    - + Remove stopwords
    - + remove empty rows (for combined_text) 23 empty rows 
    - + get emojis to useful text 
    - - Extract entities with FTNER model

    - - Add titles multiple times
    - + Rette dates i reddit posts (remove timestamp)
    - + Create a function, that find first and last dates
    - - Maybe look after if they are talking about shorting? 
    - + Get the sp500 data 

    - - TROUBLE: inconsistent path names

- In common words (visualization) - make it do so i takes combined_text

    # DATA
        - REDDIT post
        - US stock tickers (Only the US): https://www.kaggle.com/datasets/marketahead/all-us-stocks-tickers-company-info-logos
        - TIMESERIES From yahoo


TODO:
- Mapping function from company name to ticker:  
- 


"""

import kagglehub
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import yfinance as yf
import nltk
from nltk.corpus import stopwords
import re
import unicodedata
from unidecode import unidecode

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline



# Our files
import visualization

class DataHandler:
    """
    DataHandler is a class to handle data preprocessing,
    for all the data in the project: 
        - reddit walstreet post 
        - timeseries for stocks

    Preconditions:
        _load_timeseries_data(...) can only use locally saved timeseries
    
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
        self.reddit_data.head().to_csv('output.csv', index=False)
        self.reddit_data = self._preprocess_data()
        self.reddit_data.head().to_csv('new_output.csv', index=False)
        self.timeseries_data = self._load_timeseries_data(self.stock_list) 
        # self.sp500_data = self._load_sp500() 
        # print(self.sp500_data.head())

        self.start_date = None
        self.end_date = None
        self.output_df_to_csv('reddit_data')

        self.model_path = 'DATA\MODEL'

    def _load_reddit_data(self):
        file_path = './DATA/reddit_wsb.csv'
        df_reddit = pd.read_csv(file_path)
        df_reddit = df_reddit.drop(['id', 'url', 'created'], axis=1)
        return df_reddit

    def _save_sp500(self, START, END):
        # Fetch data for the S&P 500 index (^GSPC)
        sp500 = yf.Ticker("^GSPC")
        historical_data = sp500.history(start=START, end=END)
        historical_data.to_csv('./DATA/TIMESERIES/sp500.csv')

    def _load_sp500(self):
        self._save_sp500(self.start_date, self.end_date)
        file_path = './DATA/TIMESERIES/sp500.csv'
        df = pd.read_csv(file_path)
        # df['Date'] = pd.to_datetime(df['Date']).dt.date  # Convert 'Date' to date-only format
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Drop only columns that exist in the file
        columns_to_drop = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        
        df = df.drop(existing_columns_to_drop, axis=1)
        return df


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

                # Since we are changing the dataframe we have to create the labels again
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
    
   
    def combine_title_and_body(self):
        # Concatenate 'title' and 'body' columns into a new column called 'combined_text'
        df = self.reddit_data
        df['title'] = df['title'].fillna('')
        df['body'] = df['body'].fillna('')

        df['combined_text'] = df['title'] + ' ' + df['body']  # Add a space between the two columns
        return df

    def find_min_max_date(self, df):
        self.start_date = df['timestamp'].min()
        self.last_date = df['timestamp'].max()


    def remove_stop_words_and_links(self):
        df = self.reddit_data
        stop_words = set(stopwords.words('english')) 
        additional_stopwords = {'www', 'http', 'https', 'com'}
        stop_words.update(additional_stopwords)
    
        filtered_text = []
        url_pattern = re.compile(r'https?://\S+|www\.\S+')  # Regex to match URLs
    
        for text in df['combined_text']:
            # Remove URLs
            text = url_pattern.sub('', text)
            words = text.split()
            # Remove stopwords and additional terms
            filtered_words = [word for word in words if word.lower() not in stop_words]
            filtered_text.append(" ".join(filtered_words))
        
        df['combined_text'] = filtered_text
        return df
    

    def _preprocess_data(self):
        df = self.reddit_data
        title0 = 'no_stopwords'
        title1 = 'stopwords'
        title2 = 'no_stopwords_with_emojis'

        
        
        # Combine title and body
        df = self.combine_title_and_body()
        
        # Visualize before removing stop words
        self.visualize_common_words(df, title1)

        df = self.remove_stop_words_and_links()
        
        # Visualize before emojis to text
        self.visualize_common_words(df, title0)
        
        # Remove emojis
        df['combined_text'] = df['combined_text'].apply(self.deEmojify)
        
        # Drop rows where 'combined_text' is NaN or empty after stripping whitespace
        df['combined_text'] = df['combined_text'].str.strip()  # Strip leading/trailing spaces
        df = df[df['combined_text'].notna() & (df['combined_text'] != '')].reset_index(drop=True)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date
        # print(df['timestamp'].head())
        self.find_min_max_date(df)
    
         

        
        ##### Check for any remaining empty rows #####
        # empty_count = df['combined_text'].isna().sum() + (df['combined_text'] == '').sum()
        # print(f"Number of empty rows: {empty_count}")
    
        # Visualize after preprocessing
        self.visualize_common_words(df, title2)

        #REMOVE COMMMENT LATER :)
        #TODO: Add correct model to MODEL folder
        #df = self.extractNERs()

        return df
    

    # Source: https://stackoverflow.com/questions/43797500/python-replace-unicode-emojis-with-ascii-characters
    def deEmojify(self, inputString):
        returnString = ""
    
        for character in inputString:
            try:
                character.encode("ascii")
                returnString += character
            except UnicodeEncodeError:
                replaced = unidecode(str(character))
                if replaced != '':
                    returnString += replaced
                else:
                    try:
                         returnString += "[" + unicodedata.name(character) + "]"
                    except ValueError:
                         returnString += "[x]"
        return returnString
    
    def extractNERs(self):
        df = self.reddit_data
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        df["entities"] = df["combined_text"].apply(self.extract_entities)
        return df

    def extract_entities(self, text):
        ner_results = self.ner_pipeline(text)
        entities = [{"entity": result["entity_group"], "word": result["word"]} for result in ner_results]
        return entities
        
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

    def visualize_common_words(self, df, title):
        visualization.common_words(df, title)

    def visualize_word_count(self):
        visualization.word_count_distribution(self.reddit_data)

    def visualize_score_count(self):
        visualization.score_count_distribution(self.reddit_data)

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
    
    def get_combined_data(self):
        return self.reddit_data['combined_text']

    def get_ticker_dataframe(self, ticker):
        # Check if the ticker exists in the dictionary
        if ticker in self.timeseries_data:
            return self.timeseries_data[ticker]  # Return the corresponding DataFrame
        else:
            print(f"Ticker '{ticker}' not found in the data.")
            return None

    def output_df_to_csv(self, data_name):
        file_path = './DATA/'  # Directory where the CSV will be saved
        os.makedirs(file_path, exist_ok=True)  # Ensure the directory exists
        
        # Use getattr to dynamically access the attribute by name
        data_frame = getattr(self, data_name, None)
        if isinstance(data_frame, pd.DataFrame):
            # Add 'processed_' prefix to the filename
            file_name = f"processed_{data_name}.csv"
            full_file_path = os.path.join(file_path, file_name)
            
            # Save to CSV
            data_frame.to_csv(full_file_path, index=False)
            print(f"DataFrame '{data_name}' saved to: {os.path.abspath(full_file_path)}")
        else:
            print(f"'{data_name}' is not a valid DataFrame attribute.")
        

    def debugger_helper(self):
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


def sp500(START, END):
    # Fetch data for the S&P 500 index (^GSPC)
    sp500 = yf.Ticker("^GSPC")
    historical_data = sp500.history(start=START, end=END)
    historical_data.to_csv('./DATA/TIMESERIES/sp500.csv')


def tickers_timeseries_to_csv(tickers, START, END):
        path = './DATA/TIMESERIES/'
        
        for ticker in tickers:
            stock = yf.download(ticker, start=START, end=END)
            stock.to_csv(os.path.join(path, f"{ticker}.csv"), index=True)
            print(f"Data for {ticker} saved to {path}{ticker}.csv")



# tickers = ['TSLA', 'MSFT']
# START = '2020-01-01'
# END = '2022-01-01'
# 
# # tickers_timeseries_to_csv(tickers, START, END)
# sp500(START, END)


# download_annotate()
