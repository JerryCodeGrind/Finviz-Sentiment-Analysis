import yfinance as yf
import pandas as pd

ticker = 'AMD'

df = pd.read_csv('cool.csv')

# Get yfinance data
stock = yf.Ticker(ticker)
start_date = pd.to_datetime(df['date'].iloc[0])
end_date = pd.to_datetime(df['date'].iloc[-1])
stockdata = stock.history(start=start_date, end=end_date)
price = stockdata['Close']

