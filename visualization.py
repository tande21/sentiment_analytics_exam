import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import pandas as pd

# def plot_price(df, ticker):
#     # Ensure the 'Date' column is in datetime format
#     if 'Date' in df.columns:
#         df['Date'] = pd.to_datetime(df['Date'])
#         df.set_index('Date', inplace=True)  # Set 'Date' as the index
# 
#     plt.figure(figsize=(10, 5))
#     plt.title(f"{ticker}: Adj. Close Price")
#     plt.xlabel("Date")
#     plt.ylabel("Price (USD)")
# 
#     price = df['Adj Close']
#     plt.plot(df.index, price, label="Adj Close Price", color="blue")
#     plt.legend()
#     plt.show()
# 
# 
# def plot_price(df, ticker):
#     # Ensure the 'Date' column exists
#     if 'Date' in df.columns:
#         # Convert the 'Date' column to datetime format
#         df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Handle invalid dates gracefully
# 
#         # Drop rows where 'Date' could not be converted (NaT)
#         df = df.dropna(subset=['Date'])
# 
#         # Set 'Date' as the index
#         df.set_index('Date', inplace=True)
#     else:
#         print("'Date' column not found in DataFrame. Unable to plot.")
#         return
# 
#     plt.figure(figsize=(10, 5))
#     plt.title(f"{ticker}: Adj. Close Price")
#     plt.xlabel("Date")
#     plt.ylabel("Price (USD)")
# 
#     # Check if 'Adj Close' column exists
#     if 'Adj Close' in df.columns:
#         price = df['Adj Close']
#         plt.plot(df.index, price, label="Adj Close Price", color="blue")
#         plt.legend()
#         plt.show()
#     else:
#         print("'Adj Close' column not found in DataFrame. Unable to plot.")
# 
# 
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


