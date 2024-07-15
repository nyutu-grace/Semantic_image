# filename: improved_download_plot_tesla.py

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

plt.style.use('ggplot')  # set a consistent plot style

try:
    # download the data
    data = yf.download('TSLA','2019-01-01','2020-12-31')
except Exception as e:
    print("Failed to download data due to:", str(e))
    exit()

# print the fields
print(data.head())

# validate if 'Close' column exists
if 'Close' not in data.columns:
    print("Data does not have 'Close' column.")
    exit()

# validate data in 'Close' column
if data['Close'].isnull().values.any():
    print("Data in 'Close' column contains null values.")
    exit()

if (data['Close'] < 0).any():
    print("Data in 'Close' column contains negative values.")
    exit()

# calculate daily volatility: simple return
data['simple_return'] = (data['Close'] / data['Close'].shift(1)) - 1
daily_volatility = data['simple_return'].std()
print('Daily volatility is: ', daily_volatility)

# plot the data
plt.figure(figsize=(10,6))
plt.plot(data.index, data['simple_return'], label='Tesla daily return')
plt.title('Tesla daily volatility')
plt.xlabel('Date')
plt.ylabel('Daily return')
plt.legend()

# save the plot
plt.savefig('daily_volatility.png')