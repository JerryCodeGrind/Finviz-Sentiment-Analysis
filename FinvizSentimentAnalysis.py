from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf

finviz_url = 'https://finviz.com/quote.ashx?t='

ticker = 'AMD'

news_tables = {}

url = f'{finviz_url}{ticker}'
req = Request(url=url, headers = {"User-Agent": "Insert User-Agent"})
response = urlopen(req)
html = BeautifulSoup(response, 'html')

news_table = html.find(id = 'news-table') # Makes a table object 
news_tables[ticker] = news_table # Adds the table object into the dictionary with the index as that ticker symbole

parsed_data = []
    
for row in news_table.findAll('tr'):
    title = row.a.text

    # Some timestamps have dates
    date_data = row.td.text.split() # Splits text based on spaces (between jun 18-20 and 09:28PM)
    # date_data is a LIST of strings

    if len(date_data) == 1:
        time = date_data[0]
    else:
        date = date_data[0]
        time = date_data[1]

    parsed_data.append([ticker, date, time, title])

df = pd.DataFrame(parsed_data, columns = ['ticker', 'date', 'time', 'title'])

# Replace 'Today' with today's date before converting
df['date'] = df['date'].replace('Today', datetime.now().strftime('%b-%d-%y'))
df['data'] = pd.to_datetime(df['date']).dt.date

# Load the finbert model
pipe = pipeline('text-classification', model = 'ProsusAI/finbert')

# Set the score column as the sentiment score from the finbert of each article
df['score'] = df['title'].apply(lambda title: pipe(title)[0]['score'])

# Simply use groupby and mean without unstack
new_df = df.groupby('date')['score'].mean()

# Get yfinance data
stock = yf.Ticker(ticker)
start_date = pd.to_datetime(df['date'].iloc[-1])
end_date = pd.to_datetime(df['date'].iloc[0])
stockdata = stock.history(start=start_date, end=end_date)
price = stockdata['Close']

# Convert new_df index to datetime for proper alignment
new_df.index = pd.to_datetime(new_df.index)

# Calculate exponential moving averages
# Using spans of 3 and 7 days for short and medium-term trends
sentiment_ema_short = new_df.ewm(span=3, adjust=False).mean()
price_ema_short = price.ewm(span=3, adjust=False).mean()

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot sentiment scores and EMAs
ax1.plot(new_df.index, new_df.values, color='blue', label='Sentiment Score', alpha=0.6)
ax1.plot(sentiment_ema_short.index, sentiment_ema_short.values, color='red', label='3-Day EMA', linewidth=2)
ax1.set_title('Stock Sentiment Over Time')
ax1.set_ylabel('Sentiment Score')
ax1.legend()
ax1.grid(True)

# Plot stock prices and EMAs
ax2.plot(price.index, price.values, color='green', label='Stock Price', alpha=0.6)
ax2.plot(price_ema_short.index, price_ema_short.values, color='orange', label='3-Day EMA', linewidth=2)
ax2.set_title(f'{ticker} Stock Price')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price ($)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Optional: Print correlation coefficient
correlation = new_df.reindex(price.index).corr(price)
print(f"Correlation coefficient between sentiment and price: {correlation:.2f}")
