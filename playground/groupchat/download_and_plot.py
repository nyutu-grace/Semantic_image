# filename: download_and_plot.py

import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import StringIO

# download the data
url = "https://example.com/content/TSLA.csv"  # replace with your URL
response = requests.get(url)
data = response.text

# read the data into a pandas dataframe
df = pd.read_csv(StringIO(data))

# print the fields
print(df.head())

# calculate daily volatility: simple return
df['simple_return'] = (df['Close'] / df['Close'].shift(1)) - 1
daily_volatility = df['simple_return'].std()
print('Daily volatility is: ', daily_volatility)

# plot the data
plt.figure(figsize=(10,6))
plt.plot(df.index, df['simple_return'], label='Tesla daily return')
plt.title('Tesla daily volatility')
plt.xlabel('Date')
plt.ylabel('Daily return')
plt.legend()

# save the plot
plt.savefig('daily_volatility.png')