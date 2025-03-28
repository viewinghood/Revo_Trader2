### test tries to handle the following errors:
# 1 Failed download:
# ['AAPL']: JSONDecodeError('Expecting value: line 1 column 1 (char 0)')
# Failed to get ticker 'AAPL' reason: Expecting value: line 1 column 1 (char 0)

## Error went away with the following code:
# pip uninstall yfinance
# pip install yfinance --upgrade --no-cache-dir

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import pandas_ta as ta
import requests
from bs4 import BeautifulSoup

def get_stock_data(symbol: str, period: str, interval: str, max_retries: int = 3) -> pd.DataFrame:
    """Helper function to download stock data with retries"""
    for attempt in range(max_retries):
        try:
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            # df = yf.download('AAPL', period='max', interval='1d')
            if not data.empty:
                return data
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                raise Exception(f"Failed to download data for {symbol} after {max_retries} attempts: {str(e)}")
            continue
    return pd.DataFrame()  # Return empty DataFrame if all attempts fail

def main():
    start =  datetime(2025, 1, 1, tzinfo=timezone.utc) #'2025-01-01'
    end = datetime.now(tz=timezone.utc) #'2025-03-28' 
    symbol = 'AAPL'
    interval = '1d'
    period = 'max'
    #df = yf.download('AAPL', period='max', interval='1d')
    df = yf.download('AAPL', start=start, end=end,  interval=interval)
    print("Try to print Apple Stock data:",df)
    # data = get_stock_data(symbol, period, interval)
    # print("Try to print Apple Stock data:",data)

if __name__ == "__main__":
    main()
