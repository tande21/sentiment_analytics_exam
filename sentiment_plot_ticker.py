import pandas as pd
import matplotlib
# matplotlib.use('Agg')  # If you want interactive plots, remove this line
import matplotlib.pyplot as plt
import ast

import data_preprocessing
from helper import load_data

CLASS_NAME = 'DataHandler' 
DATAHANDLER = data_preprocessing.loadDataHandler(CLASS_NAME)
GME_TIMESERIES = DATAHANDLER.get_ticker_dataframe('AMZ')

COMPLETED_DF_SENTIMENT = load_data('./DATA/COMPLETED_DF_SENTIMENT.csv')

def extract_label(sentiment_str):
    try:
        sentiment_list = ast.literal_eval(sentiment_str)  # Safely parse the string to a list of dictionaries
        if isinstance(sentiment_list, list) and 'label' in sentiment_list[0]:
            return sentiment_list[0]['label']
    except (ValueError, SyntaxError):
        return None
    return None

def map_sentiment_to_timeseries(timeseries, sentiment_df, filename='GME_plot'):
    """
    Map sentiment data to a timeseries, visualize it, and save the plot.
    """
    if 'Adj Close' not in timeseries.columns:
        raise ValueError("The timeseries DataFrame must have an 'Adj Close' column.")

    # Convert timestamps to datetime and ensure the 'date' column matches the dtype of the timeseries index
    sentiment_df['date'] = pd.to_datetime(sentiment_df['timestamp']).dt.date
    # Convert to full datetime64[ns] to match timeseries index type
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

    sentiment_summary = sentiment_df.groupby('date')['Sentiment'].apply(list).reset_index()

    # Ensure timeseries index is datetime64[ns] as well
    # If it's already datetime64[ns], this step is not needed. If it's just a date, convert:
    # timeseries.index = pd.to_datetime(timeseries.index) # Uncomment if needed

    # Filter timeseries to only include dates in the sentiment data
    # Now both sides should be datetime64[ns]
    timeseries = timeseries[timeseries.index.isin(sentiment_summary['date'])]

    # Set the index of sentiment_summary to 'date' for easy aligning
    sentiment_summary.set_index('date', inplace=True)

    # Use .loc to avoid SettingWithCopyWarning
    timeseries = timeseries.copy()  # Make a copy to ensure assignments are safe
    timeseries.loc[:, 'Sentiment'] = sentiment_summary['Sentiment']

    # Assign colors
    timeseries.loc[:, 'Color'] = timeseries['Sentiment'].apply(
        lambda sentiments: 'red' if 'NEGATIVE' in sentiments else 'green'
    )

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(timeseries.index, timeseries['Adj Close'], c=timeseries['Color'], label='Sentiment')
    plt.plot(timeseries.index, timeseries['Adj Close'], color='black', alpha=0.5, linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.title('Timeseries with Sentiment Mapping')
    plt.legend()

    # If running in a non-interactive environment, plt.show() might not display anything, but no harm in calling it
    plt.show()

    save_path = f'./{filename}.png'
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

    return timeseries

if __name__ == "__main__":
    # Convert sentiment column
    COMPLETED_DF_SENTIMENT['Sentiment'] = COMPLETED_DF_SENTIMENT['Sentiment'].apply(extract_label)

    mapped_df = map_sentiment_to_timeseries(GME_TIMESERIES, COMPLETED_DF_SENTIMENT, filename='GME_plot')
    print(mapped_df)

