from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

finviz_url = 'https://finviz.com/quote.ashx?t='

tickers = ['AAPL', 'AMZN', 'AMD']

news_tables = {}

for ticker in tickers:
    url = f'{finviz_url}{ticker}'
    req = Request(url=url, headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"})
    response = urlopen(req)
    html = BeautifulSoup(response, 'html')

    news_table = html.find(id = 'news-table') # Makes a table object 
    news_tables[ticker] = news_table # Adds the table object into the dictionary with the index as that ticker symbole

parsed_data = []

for ticker, news_table in news_tables.items(): # Takes the key and value for each item in news tables dictionary
    
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

new_df = df.groupby(['date', 'ticker'])['score'].mean().unstack()  # Switch order of grouping
# Now 'date' is the index and 'ticker' creates the columns

print(new_df)

new_df.plot(kind='line', figsize=(10, 8))
plt.title('Stock Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')

plt.tight_layout()
plt.show()