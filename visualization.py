"""
file: visualization.py
purpose: These are the functions we use to visualize,
    doing the project for timeseries and reddit post.
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


def plot_price(df, ticker):
    # Print column names to verify
    print(f"Columns in DataFrame: {df.columns}")
    
    # If there's no 'Date' column, assume the DataFrame's index contains the date.
    # Convert the index to datetime if it's not already.
    if not isinstance(df.index, pd.DatetimeIndex):
        # Attempt to convert the index to a DatetimeIndex
        df.index = pd.to_datetime(df.index, errors='coerce')
        # Drop rows where the index couldn't be converted to datetime
        df = df[df.index.notnull()]
    
    # Now 'df.index' should be a DatetimeIndex representing the dates.
    # Check if 'Adj Close' column exists
    adj_close_columns = [col for col in df.columns if col.lower() == 'adj close']
    if adj_close_columns:
        price_col = adj_close_columns[0]
    else:
        print(f"No 'Adj Close' column found in DataFrame for {ticker}")
        return

    plt.figure(figsize=(10, 5))
    plt.title(f"{ticker}: Adj. Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    
    # Plot using the DatetimeIndex
    plt.plot(df.index, df[price_col], label="Adj Close Price", color="blue")
    plt.legend()
    plt.show()


def common_words(df_format, title):
    
    plot_path = './DATA/PLOTS/'
    path = plot_path + title + '_common_words'

    # Handle NaN values by replacing them with an empty string
    df_format['combined_text'] = df_format['combined_text'].replace(np.nan, '', regex=True)
    # Use CountVectorizer to tokenize and count word frequencies
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df_format['combined_text'])
    
    # Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Sum the occurrences of each word across all sentences
    word_frequencies = Counter(dict(zip(feature_names, X.sum(axis=0).A1)))
    
    # Display the most common words and their frequencies
    most_common_words = word_frequencies.most_common(10)
    most_common_words = most_common_words[0:10]

    plt.figure(figsize=(12, 8))
    plt.bar(*zip(*most_common_words))
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10 Most Common Words')
    plt.savefig(path)
    plt.close()
    # plt.show()

def count_words(text):
    if pd.isna(text):  # Check for NaN values
        return 0
    words = str(text).split()  # Convert to string and split
    return len(words)


def word_count_distribution(df):
    plot_path = './DATA/PLOTS/' 
    path = plot_path + 'Reddit_word_distribution'


    # Add a word count column
    df['Word_Count'] = df['combined_text'].apply(count_words)

    # Exclude outliers: Cap the word count at a sensible maximum
    max_word_count = 200  # Adjust based on your data
    filtered_df = df[df['Word_Count'] <= max_word_count]

    # Recalculate percentiles after filtering
    percentiles = filtered_df['Word_Count'].describe(percentiles=[0.25, 0.5, 0.75])

    # Build word count distribution
    word_count_dict = {}
    for count in filtered_df['Word_Count']:
        if count in word_count_dict:
            word_count_dict[count] += 1
        else:
            word_count_dict[count] = 1

    # Convert dictionary items to lists for plotting
    word_counts = list(word_count_dict.keys())
    row_counts = list(word_count_dict.values())

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(12, 8))

    # Table for percentiles
    table_data = [
        ['25%', f"{percentiles['25%']:.2f}"],
        ['50%', f"{percentiles['50%']:.2f}"],
        ['75%', f"{percentiles['75%']:.2f}"]
    ]
    table = ax2.table(cellText=table_data, loc='center', 
                      colLabels=['Percentile', 'Value'], cellLoc='center', colColours=['#f0f0f0'] * 2)

    ax2.axis('off')  # Hide axes for the table

    # Plotting the bar chart
    ax1.set_xlabel('Word Count')
    ax1.set_ylabel('Number of Rows')
    ax1.set_title('Reddit Word Count Distribution in Rows')
    ax1.bar(word_counts, row_counts, color='blue')

    # Save the plot
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def score_count_distribution(df):

    plot_path = './DATA/PLOTS/' 
    path = plot_path + 'Reddit_score_distribution'
    print(df.columns)

    # Add a 'Score' column (if not already present)

    # Exclude outliers: Cap the score at a sensible maximum
    max_score = 350  # Adjust based on your data
    filtered_df = df[df['score'] <= max_score]

    # Recalculate percentiles after filtering
    percentiles = filtered_df['score'].describe(percentiles=[0.25, 0.5, 0.75])

    # Build score distribution
    score_count_dict = {}
    for score in filtered_df['score']:
        if score in score_count_dict:
            score_count_dict[score] += 1
        else:
            score_count_dict[score] = 1

    # Convert ictionary items to lists for plotting
    scores = list(score_count_dict.keys())
    row_counts = list(score_count_dict.values())

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(12, 8))

    # Table for percentiles
    table_data = [
        ['25%', f"{percentiles['25%']:.2f}"],
        ['50%', f"{percentiles['50%']:.2f}"],
        ['75%', f"{percentiles['75%']:.2f}"]
    ]
    table = ax2.table(cellText=table_data, loc='center', 
                      colLabels=['Percentile', 'Value'], cellLoc='center', colColours=['#f0f0f0'] * 2)

    ax2.axis('off')  # Hide axes for the table

    # Plotting the bar chart
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Number of Rows')
    ax1.set_title('Reddit Score Distribution')
    ax1.bar(scores, row_counts, color='blue')

    # Save the plot
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
