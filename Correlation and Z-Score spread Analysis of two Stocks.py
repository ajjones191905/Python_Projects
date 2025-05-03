from polygon.rest import RESTClient
import os
import pandas as pd 
from datetime import datetime
import sqlite3
import matplotlib.dates as mdates
import matplotlib.pyplot as plt 

API_Key = "rPmnwPI_Bj9Fc7LlyGp6ErFqDuQ7_PmW"  
client = RESTClient(API_Key)

def get_min_data(symbol, start_date, end_date):
    start = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    data = []
    try:
        response = client.get_aggs(symbol, 1, "minute", start, end)
        if response:
            for agg in response:
                timestamp = datetime.fromtimestamp(agg.timestamp / 1000)
                close = agg.close
                if timestamp.hour >= 9 and timestamp.hour < 16:
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

def store_data_sql(df, db_name="stocks.db"):
    if df.empty:
        print("No data to store.")
        return

    conn = sqlite3.connect(db_name)
    conn.execute('''CREATE TABLE IF NOT EXISTS stock_data (
        symbol TEXT, 
        timestamp DATETIME, 
        close FLOAT, 
        PRIMARY KEY (symbol, timestamp)
    );''')

    for index, row in df.iterrows():
        timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        conn.execute('''INSERT OR REPLACE INTO stock_data (symbol, timestamp, close) 
                        VALUES (?, ?, ?)''', 
                        (row['symbol'], timestamp_str, row['close']))
    
    conn.commit()
    conn.close()

def calc_corr(stock1, stock2, db_name="stocks.db"):
    conn = sqlite3.connect(db_name)
    query = f"""
    SELECT timestamp, symbol, close FROM stock_data
    WHERE symbol = '{stock1}' OR symbol = '{stock2}'
    """
    df = pd.read_sql_query(query, conn)
    conn.close())

    if df[df['symbol'] == stock1].empty or df[df['symbol'] == stock2].empty:
        print(f"Missing data for either {stock1} or {stock2}. Cannot calculate correlation.")
        return None, None

    pivot_df = df.pivot(index="timestamp", columns="symbol", values="close")
    pivot_df = pivot_df.dropna()

    if len(pivot_df) > 1:
        correlation = pivot_df.corr().iloc[0, 1]
        return correlation, pivot_df
    else:
        print(f"Not enough data for correlation calculation between {stock1} and {stock2}.")
        return None, None


def z_score_spread(pivot_df): 
    spread = pivot_df.iloc[:, 0] - pivot_df.iloc[:, 1]

    z_score = (spread - spread.mean()) / spread.std()
    return z_score

def plot_stock_data_and_zscore(df1, df2, symbol1, symbol2):
    df1['hour'] = df1['timestamp'].dt.hour
    df1['minute'] = df1['timestamp'].dt.minute
    df2['hour'] = df2['timestamp'].dt.hour
    df2['minute'] = df2['timestamp'].dt.minute
    
    df1_filtered = df1[((df1['hour'] > 9) | ((df1['hour'] == 9) & (df1['minute'] >= 30))) & (df1['hour'] < 16)]
    df2_filtered = df2[((df2['hour'] > 9) | ((df2['hour'] == 9) & (df2['minute'] >= 30))) & (df2['hour'] < 16)]

    df1_filtered['percentage_change'] = (df1_filtered['close'] - df1_filtered['close'].iloc[0]) / df1_filtered['close'].iloc[0] * 100
    df2_filtered['percentage_change'] = (df2_filtered['close'] - df2_filtered['close'].iloc[0]) / df2_filtered['close'].iloc[0] * 100

    df_combined = pd.merge(df1_filtered[['timestamp', 'percentage_change']], 
                            df2_filtered[['timestamp', 'percentage_change']], 
                            on='timestamp', suffixes=(f'_{symbol1}', f'_{symbol2}'))

    df_combined['date'] = df_combined['timestamp'].dt.date
    unique_dates = df_combined['date'].unique()

    combined_data = []
    last_timestamp = None

    for date in unique_dates:
        session_data = df_combined[df_combined['date'] == date]
        
        if last_timestamp is not None:
            session_data['timestamp'] = session_data['timestamp'] + (last_timestamp - session_data['timestamp'].iloc[0])

        combined_data.append(session_data)
        
        last_timestamp = session_data['timestamp'].iloc[-1]

    df_combined_continuous = pd.concat(combined_data)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

   
    ax1.plot(df_combined_continuous['timestamp'], df_combined_continuous[f'percentage_change_{symbol1}'], label=f'{symbol1} Percentage Change', color='blue', linestyle='-', linewidth=1)
    ax1.plot(df_combined_continuous['timestamp'], df_combined_continuous[f'percentage_change_{symbol2}'], label=f'{symbol2} Percentage Change', color='red', linestyle='-', linewidth=1)
    ax1.set_title('Stock Percentage Change Over Time')
    ax1.set_ylabel('Percentage Change (%)')
    ax1.legend(loc='upper left')

    
    spread = df_combined_continuous[f'percentage_change_{symbol1}'] - df_combined_continuous[f'percentage_change_{symbol2}']
    zscore_spread = (spread - spread.mean()) / spread.std()

    ax2.plot(df_combined_continuous['timestamp'], zscore_spread, label='Z-Score of Spread', color='green', linestyle='-', linewidth=1)
    ax2.set_title(f'Z-Score Spread Over Time which is {symbol1}-{symbol2}')
    ax2.set_ylabel('Z-Score')
    ax2.legend(loc='upper left')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__": 
    symbol1 = "NCLH"
    symbol2 = "RCL"
    start_date = "2023-8-14"
    end_date = "2025-01-27"
    
    df1 = get_min_data(symbol1, start_date, end_date)
    df2 = get_min_data(symbol2, start_date, end_date)

    store_data_sql(df1)
    store_data_sql(df2)

    correlation, pivot_df = calc_corr(symbol1, symbol2)
    if correlation is not None:
        print(f"Correlation between {symbol1} and {symbol2}: {correlation:.2f}")
        z_score = z_score_spread(pivot_df)
    else:
        print(f"Cannot calculate correlation for {symbol1} and {symbol2}.")
    
    plot_stock_data_and_zscore(df1, df2, symbol1, symbol2)
