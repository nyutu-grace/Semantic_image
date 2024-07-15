# filename: download_and_print.py

import pandas as pd
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