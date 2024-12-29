from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np

finviz_url = 'https://finviz.com/quote.ashx?t='

ticker = 'AMD'

news_tables = {}

url = f'{finviz_url}{ticker}'
req = Request(url=url, headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"})
response = urlopen(req)
html = BeautifulSoup(response, 'lxml')

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

# Combine date and time into a single datetime column
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

# Sort by datetime
df = df.sort_values('datetime')

# Wrap the model loading in a main block to avoid multiprocessing issues
if __name__ == '__main__':
    # Load the finbert model
    pipe = pipeline('text-classification', model='ProsusAI/finbert')
    
    # Calculate sentiment scores
    df['score'] = df['title'].apply(lambda title: pipe(title)[0]['score'])
    
    # Convert datetime to numeric values for regression
    df['numeric_time'] = (df['datetime'] - df['datetime'].min()).dt.total_seconds()
    
    # Prepare data for linear regression
    X = df['numeric_time'].values.reshape(-1, 1)
    y = df['score'].values
    
    # Fit linear regression
    reg = LinearRegression()
    reg.fit(X, y)
    
    # Generate points for regression line
    line_x = np.array([X.min(), X.max()]).reshape(-1, 1)
    line_y = reg.predict(line_x)
    
    # Convert numeric time back to datetime for plotting
    line_dates = [df['datetime'].min() + pd.Timedelta(seconds=x[0]) for x in line_x]

# Get stock data at same frequency
start_date = df['datetime'].min()
end_date = df['datetime'].max()
stock = yf.Ticker(ticker)
stockdata = stock.history(start=start_date, end=end_date, interval='30m')
price = stockdata['Close']

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot sentiment scores with regression line
ax1.scatter(df['datetime'], df['score'], color='blue', alpha=0.3, label='Raw Sentiment')
ax1.plot(line_dates, line_y, color='red', label=f'Trend Line (slope: {reg.coef_[0]:.2e})', linestyle='--')
ax1.set_title('Intraday Stock Sentiment')
ax1.set_ylabel('Sentiment Score')
ax1.set_ylim(0, 1)
ax1.legend()
ax1.grid(True)

# Plot stock prices
smooth_price = price.resample('1H').mean().interpolate(method='cubic')
ax2.plot(smooth_price.index, smooth_price.values, color='green', label='Stock Price', alpha=0.8)
ax2.set_title(f'{ticker} Stock Price')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price ($)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Calculate and print correlation statistics
correlation = df['score'].corr(df.set_index('datetime')['numeric_time'])
print(f"Correlation coefficient between sentiment and time: {correlation:.2f}")
print(f"Regression slope: {reg.coef_[0]:.2e}")
print(f"R-squared score: {reg.score(X, y):.2f}")