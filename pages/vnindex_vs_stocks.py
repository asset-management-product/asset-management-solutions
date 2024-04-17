import streamlit as st

import requests
from io import BytesIO
#importing packages
import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt

import scipy.optimize as spop
import plotly.graph_objs as go
from io import StringIO

st.write('Hello')

#ohld means open, high, low, close
# Correct the function to load Excel files
def load_data(url):
    # Make a GET request to fetch the raw Excel content
    response = requests.get(url)
    response.raise_for_status()  # This will raise an HTTPError if the request returned an unsuccessful status code.
    
    # Use BytesIO to convert the binary content into a file-like object so pd.read_excel can read it
    data = BytesIO(response.content)
    
    # Read the data into a pandas DataFrame
    return pd.read_excel(data)

# URL pointing to the Excel file
file_path_index = 'https://github.com/penthousecompany/master/raw/main/curated/curated_vnindex.xlsx'

# Load the data
df_index = load_data(file_path_index)
df_index=df_index.iloc[:,0].str.split(',', expand=True)

# Define column names based on the provided information
column_names = [
    "Ngay", "GiaDieuChinh", "Giá Đóng Cửa", "Thay Đổi", "Khối Lượng Khớp Lệnh", 
    "Giá Trị Khớp Lệnh", "KL Thỏa Thuận", "GT Thỏa Thuận", "Giá Mở Cửa", 
    "Giá Cao Nhất", "Giá Thấp Nhất"
]

# Assign the column names to the dataframe
df_index.columns = column_names
df_index=df_index[['Ngay','GiaDieuChinh']]
#ohld means open, high, low, close
# URL to the raw CSV file on GitHub
url = 'https://raw.githubusercontent.com/penthousecompany/master/main/curated/curated_stock_ohlc.csv'

# Make a GET request to fetch the raw CSV content
response = requests.get(url)
response.raise_for_status()  # This will raise an HTTPError if the request returned an unsuccessful status code.

# Use StringIO to convert the text content into a file-like object so pd.read_csv can read it
data = StringIO(response.text)

# Read the data into a pandas DataFrame
data = pd.read_csv(data)

#modify data
if "datime" in data.columns:
    data=data.drop('datime',axis=1)
else:
    pass
data['time'] =  pd.to_datetime(data['time'])
data_agg = data.groupby(['time', 'ticker']).mean().reset_index()
pivot_data = data_agg.pivot(index='time', columns='ticker', values='close')

# Define tickers and date range
tickers = ['VNM','VCB','MWG','HPG','PNJ','FPT']
pivot_data=pivot_data[tickers].sort_index(ascending=True)

start = pd.to_datetime('2022-01-01')
end = pd.to_datetime('2023-12-09')

prices_df=pivot_data.loc[start:end]

df_index['Ngay'] = pd.to_datetime(df_index['Ngay'],format='%d/%m/%Y')
df_index_filtered = df_index[(df_index['Ngay'] >= start) & (df_index['Ngay'] <= end)]

# Merge selected_data_between_dates with df_index_filtered based on the 'Ngay' column
merged_data = pd.merge(prices_df, df_index_filtered, left_index=True, right_on='Ngay', how='left')
merged_data.set_index('Ngay', inplace=True)
merged_data = merged_data.rename(columns={"GiaDieuChinh": "VNINDEX"})
merged_data.dropna(inplace=True)
st.write(merged_data)

data_normalized = merged_data #/ merged_data.iloc[0] * 100

# Plot the normalized data as lines
data_normalized.plot.line(figsize=(20, 6))

# Create a Plotly figure
fig = go.Figure()

# Add traces (lines) for each ticker
for ticker in data_normalized.columns:
    fig.add_trace(go.Scatter(
        x=data_normalized.index,
        y=data_normalized[ticker],
        mode='lines',
        name=ticker
    ))

    # Annotate the last point of each line with the ticker
    last_price = data_normalized[ticker].iloc[-5]
    fig.add_annotation(
        x=data_normalized.index[-1],
        y=last_price,
        text=ticker,
        showarrow=True,
        arrowhead=1,
        ax=40,  # Adjusts the position of the annotation arrow
        ay=0
    )

# Update layout (similar to plt.title, plt.xlabel, plt.ylabel)
fig.update_layout(
    title='Normalized Stock Prices Over Time',
    xaxis_title='Date',
    yaxis_title='Normalized Stock Price (First Day = 100)',
    legend_title='Ticker'
)

# Display the figure in Streamlit
st.plotly_chart(fig)

returns_df = merged_data.pct_change()[1:]
# Remove rows where VNINDEX return is 0
returns_df = returns_df[returns_df['VNINDEX'] != 0]

# Assuming "returns_df" is your dataframe with daily returns
# Assuming "merged_data" is your dataframe with stock prices and VNINDEX

# Define the window size
window_size = 365

# Create an empty dataframe to store beta values
beta_df = pd.DataFrame(index=returns_df.index)

# Calculate beta for each stock
for stock in returns_df.columns[:-1]:  # Exclude VNINDEX
    beta_values = []

    for i in range(window_size-1, len(returns_df)):
        # Ensure that we have enough data points in the window
        if i - window_size < 0:
            continue

        # Select the past 365 days of data
        stock_returns = returns_df[stock].iloc[i - window_size:i]
        market_returns = returns_df['VNINDEX'].iloc[i - window_size:i]

        # Calculate covariance and variance
        covar = np.cov(stock_returns, market_returns)[0, 1]
        var_market = np.var(market_returns)

        # Calculate beta
        beta = covar / var_market
        beta_values.append(beta)

    # Extend beta values with NaN for the initial days where there's not enough data
    beta_values = [np.nan] * (window_size - 1) + beta_values
    beta_values.append(beta_values[-1])
    beta_df[stock + '_beta'] = beta_values

# Assuming 'beta_df' and 'data_normalized' are your DataFrame with the beta values
    
import streamlit as st
import plotly.graph_objs as go
import pandas as pd

# Assuming 'beta_df' is your DataFrame

# Create a Plotly figure
fig = go.Figure()

# Add traces (lines) for each ticker
for ticker in beta_df.columns:
    fig.add_trace(go.Scatter(
        x=beta_df.index,
        y=beta_df[ticker],
        mode='lines',
        name=ticker
    ))

    # Annotate the last point of each line with the ticker
    last_price = beta_df[ticker].iloc[-5]
    fig.add_annotation(
        x=beta_df.index[-1],
        y=last_price,
        text=ticker,
        showarrow=True,
        arrowhead=1,
        ax=70,  # Adjusts the position of the annotation arrow
        ay=0
    )

# Update layout (similar to plt.title, plt.xlabel, plt.ylabel)
fig.update_layout(
    title='Beta Over Time',
    xaxis_title='Date',
    yaxis_title='Beta',
    legend_title='Ticker'
)

# Display the figure in Streamlit
st.plotly_chart(fig)
