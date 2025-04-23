from polygon.rest import RESTClient
import os
import pandas as pd 
from datetime import datetime
import sqlite3
import matplotlib.dates as mdates
import matplotlib.pyplot as plt 

API_Key = "rPmnwPI_Bj9Fc7LlyGp6ErFqDuQ7_PmW"  # Replace with your actual API key

# Initialize the RESTClient with your API key
client = RESTClient(API_Key)

# Function to fetch minute-level data from Polygon.io
def get_min_data(symbol, start_date, end_date):
    start = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    # Get the minute-level data
    data = []
    try:
        # Fetch data for the given symbol and time range
        response = client.get_aggs(symbol, 1, "minute", start, end)
        if response:
            #print(f"Received data for {symbol}:")
            for agg in response:
                timestamp = datetime.fromtimestamp(agg.timestamp / 1000)
                close = agg.close
                # Filter for market hours: from 9:30 AM to 4:00 PM ET
                if timestamp.hour >= 9 and timestamp.hour < 16:
                    #print(f"Timestamp: {timestamp}, Close: {close}")
                    data.append({
                        "symbol": symbol, 
                        "timestamp": timestamp, 
                        "close": close, 
                    })
        else:
            print(f"No data found for {symbol} between {start_date} and {end_date}")
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
    
    if not data:
        print(f"Warning: No valid data fetched for {symbol} in the given range.")
    
    return pd.DataFrame(data)

# Function to store data in an SQLite database
def store_data_sql(df, db_name="stocks.db"):
    if df.empty:
        print("No data to store.")
        return
    
    conn = sqlite3.connect(db_name)
    # Do not drop the table; just insert the new data
    conn.execute('''CREATE TABLE IF NOT EXISTS stock_data (
        symbol TEXT, 
        timestamp DATETIME, 
        close FLOAT, 
        PRIMARY KEY (symbol, timestamp)
    );''')

    # Insert data
    for index, row in df.iterrows():
        timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        conn.execute('''INSERT OR REPLACE INTO stock_data (symbol, timestamp, close) 
                        VALUES (?, ?, ?)''', 
                        (row['symbol'], timestamp_str, row['close']))
    
    conn.commit()
    conn.close()
    #print(f"Data for {df['symbol'].iloc[0]} stored successfully.")

# Function to calculate correlation between two stocks
def calc_corr(stock1, stock2, db_name="stocks.db"):
    conn = sqlite3.connect(db_name)

    query = f"""
    SELECT timestamp, symbol, close FROM stock_data
    WHERE symbol = '{stock1}' OR symbol = '{stock2}'
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Print the first few rows of data for each stock
    #print(f"Data for {stock1}:")
    #print(df[df['symbol'] == stock1].head())

    #print(f"Data for {stock2}:")
    #print(df[df['symbol'] == stock2].head())

    if df[df['symbol'] == stock1].empty or df[df['symbol'] == stock2].empty:
        print(f"Missing data for either {stock1} or {stock2}. Cannot calculate correlation.")
        return None, None

    # Pivot data and compute correlation
    pivot_df = df.pivot(index="timestamp", columns="symbol", values="close")
    #print("Pivoted DataFrame:")
    #print(pivot_df.head())

    pivot_df = pivot_df.dropna()  # Drop any rows with NaN values

    if len(pivot_df) > 1:
        correlation = pivot_df.corr().iloc[0, 1]
        return correlation, pivot_df
    else:
        print(f"Not enough data for correlation calculation between {stock1} and {stock2}.")
        return None, None

# Function to calculate Z-score spread
def z_score_spread(pivot_df): 
    spread = pivot_df.iloc[:, 0] - pivot_df.iloc[:, 1]

    # Calculate z-score
    z_score = (spread - spread.mean()) / spread.std()
    return z_score

def plot_stock_data_and_zscore(df1, df2, symbol1, symbol2):
    # Filter data for each day, keeping only the time between 9:30 AM and 4:00 PM
    df1['hour'] = df1['timestamp'].dt.hour
    df1['minute'] = df1['timestamp'].dt.minute
    df2['hour'] = df2['timestamp'].dt.hour
    df2['minute'] = df2['timestamp'].dt.minute
    
    # Filter for market hours: between 9:30 AM and 4:00 PM (excluding weekends/closed market days)
    df1_filtered = df1[((df1['hour'] > 9) | ((df1['hour'] == 9) & (df1['minute'] >= 30))) & (df1['hour'] < 16)]
    df2_filtered = df2[((df2['hour'] > 9) | ((df2['hour'] == 9) & (df2['minute'] >= 30))) & (df2['hour'] < 16)]

    # Calculate percentage change from day 1 for both stocks
    df1_filtered['percentage_change'] = (df1_filtered['close'] - df1_filtered['close'].iloc[0]) / df1_filtered['close'].iloc[0] * 100
    df2_filtered['percentage_change'] = (df2_filtered['close'] - df2_filtered['close'].iloc[0]) / df2_filtered['close'].iloc[0] * 100

    # Merge the dataframes for combined plotting
    df_combined = pd.merge(df1_filtered[['timestamp', 'percentage_change']], 
                            df2_filtered[['timestamp', 'percentage_change']], 
                            on='timestamp', suffixes=(f'_{symbol1}', f'_{symbol2}'))

    # Remove the market gap between close and open by adjusting the x-axis
    df_combined['date'] = df_combined['timestamp'].dt.date
    unique_dates = df_combined['date'].unique()

    # List to store the combined data
    combined_data = []
    last_timestamp = None

    # For each day, reset the time and concatenate the data
    for date in unique_dates:
        # Filter data for each day
        session_data = df_combined[df_combined['date'] == date]
        
        # Adjust the timestamp to make it continuous (no gap between sessions)
        if last_timestamp is not None:
            # Shift the timestamp by adding the difference between sessions
            session_data['timestamp'] = session_data['timestamp'] + (last_timestamp - session_data['timestamp'].iloc[0])

        # Append session data to the combined data
        combined_data.append(session_data)
        
        # Update the last timestamp to the last point in this session
        last_timestamp = session_data['timestamp'].iloc[-1]

    # Concatenate all days into one dataframe
    df_combined_continuous = pd.concat(combined_data)

    # Plotting the stock percentage changes without dots (just lines), ensuring no line after 4:00 PM
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot Stock Percentage Change with lines only (no dots)
    ax1.plot(df_combined_continuous['timestamp'], df_combined_continuous[f'percentage_change_{symbol1}'], label=f'{symbol1} Percentage Change', color='blue', linestyle='-', linewidth=1)
    ax1.plot(df_combined_continuous['timestamp'], df_combined_continuous[f'percentage_change_{symbol2}'], label=f'{symbol2} Percentage Change', color='red', linestyle='-', linewidth=1)
    ax1.set_title('Stock Percentage Change Over Time')
    ax1.set_ylabel('Percentage Change (%)')
    ax1.legend(loc='upper left')

    # Plotting the Z-Score of the spread (difference between the two stock prices)
    spread = df_combined_continuous[f'percentage_change_{symbol1}'] - df_combined_continuous[f'percentage_change_{symbol2}']
    zscore_spread = (spread - spread.mean()) / spread.std()

    ax2.plot(df_combined_continuous['timestamp'], zscore_spread, label='Z-Score of Spread', color='green', linestyle='-', linewidth=1)
    ax2.set_title(f'Z-Score Spread Over Time which is {symbol1}-{symbol2}')
    ax2.set_ylabel('Z-Score')
    ax2.legend(loc='upper left')

    # Rotate x-axis labels for readability and apply tight layout
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




# Main function to run the script
if __name__ == "__main__": 
    # Define the stocks and range
    symbol1 = "NCLH"
    symbol2 = "RCL"
    start_date = "2023-8-14"
    end_date = "2025-01-27"
    
    # Fetch data for both stocks
    df1 = get_min_data(symbol1, start_date, end_date)
    df2 = get_min_data(symbol2, start_date, end_date)

    # Store the data in the database
    store_data_sql(df1)
    store_data_sql(df2)

    # Check correlation if data exists for both stocks
    correlation, pivot_df = calc_corr(symbol1, symbol2)
    if correlation is not None:
        print(f"Correlation between {symbol1} and {symbol2}: {correlation:.2f}")
        z_score = z_score_spread(pivot_df)
    else:
        print(f"Cannot calculate correlation for {symbol1} and {symbol2}.")
    
    print("DONE now matplotlib")
    plot_stock_data_and_zscore(df1, df2, symbol1, symbol2)
