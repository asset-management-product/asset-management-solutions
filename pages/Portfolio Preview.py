import streamlit as st
import pandas as pd
import numpy as np

import requests
from io import StringIO


# Function to get the latest price of a ticker
def get_latest_price(ticker, stock_data):
    latest_data = stock_data[stock_data['ticker'] == ticker].sort_values(by='time', ascending=False)
    return latest_data.iloc[0]['close'] if not latest_data.empty else None

# Function to read initial portfolio data
def read_portfolio(cash):
    # Example portfolio data
    portfolio= pd.DataFrame({
        'ticker': ['MIG','FPT','PNJ','MBB','DCM','BIC','DQC','GDT'],
        'Shares': [500, 200,400,1000,400,500,500,400],
        })
    # Add a new column 'Price' by applying get_latest_price function for each ticker
    portfolio['Price'] = portfolio['ticker'].apply(lambda x: get_latest_price(x, stock_data))
    portfolio['Value'] = portfolio['Shares']*portfolio['Price']
    
    total_value=portfolio['Value'].sum()+cash
    portfolio['Contribution'] = (portfolio['Value']/total_value*100).round(2).astype(str) + '%'

        # Add cash as a row
    cash_row = pd.DataFrame({
        'ticker': ['Cash'],
        'Shares': [np.nan],
        'Price': [np.nan],
        'Value': [cash],
        'Contribution': [(cash / total_value * 100).round(2).astype(str) + '%']
    })

    total_row= pd.DataFrame({
        'ticker':['Total'],
        'Shares':[np.nan],
        'Price':[np.nan],
        'Value':[total_value],
        'Contribution': "100%"

    })

    portfolio = pd.concat([portfolio, cash_row], ignore_index=True)
    portfolio = pd.concat([portfolio, total_row], ignore_index=True)
    return portfolio,cash,total_value
# Update portfolio based on the orders
def update_portfolio(orders, initial_portfolio,cash, stock_data):
    portfolio = initial_portfolio.copy()
    for order in orders:
        price = get_latest_price(order['ticker'], stock_data)
        order_value = price * order['Shares']
        if order['Type'] == 'Buy':
            if order['ticker'] in portfolio['ticker'].values:
                portfolio.loc[portfolio['ticker'] == order['ticker'], 'Shares'] += order['Shares']
            else:
                new_row = pd.DataFrame({'ticker': [order['ticker']], 'Shares': [order['Shares']], 'Price': [price], 'Value': [price * order['Shares']]})
                portfolio = pd.concat([portfolio, new_row], ignore_index=True)
                
            cash -= order_value
        elif order['Type'] == 'Sell':
            if order['ticker'] in portfolio['ticker'].values and portfolio.loc[portfolio['ticker'] == order['ticker'], 'Shares'].iloc[0] >= order['Shares']:
                portfolio.loc[portfolio['ticker'] == order['ticker'], 'Shares'] -= order['Shares']
                if portfolio.loc[portfolio['ticker'] == order['ticker'], 'Shares'].iloc[0] == 0:
                    portfolio = portfolio[portfolio['ticker'] != order['ticker']]
            cash += order_value
    # Recalculate the values
    portfolio['Price'] = portfolio['ticker'].apply(lambda x: get_latest_price(x, stock_data))
    portfolio['Value'] = portfolio['Shares'] * portfolio['Price']
    
    # Handle zero total value scenario
    total_value = portfolio['Value'].sum()+cash
    if total_value > 0:
        portfolio['Contribution'] = (portfolio['Value'] / total_value*100).round(2).astype(str) + '%'
    else:
        portfolio['Contribution'] = 0

    # Update cash row
    cash_index = portfolio[portfolio['ticker'] == 'Cash'].index[0]
    portfolio.loc[cash_index, 'Value'] = cash
    portfolio.loc[cash_index, 'Contribution'] = (cash / total_value * 100).round(2).astype(str) + '%'
    
    # Update Total Value
    total_index= portfolio[portfolio['ticker'] == 'Total'].index[0]
    portfolio.loc[total_index, 'Value'] = total_value
    portfolio.loc[total_index, 'Contribution'] = "100%"

    return portfolio,cash,total_value


# Load stock data
# URL to the raw CSV file on GitHub
url = 'https://raw.githubusercontent.com/penthousecompany/master/main/curated/curated_stock_ohlc.csv'

# Make a GET request to fetch the raw CSV content
response = requests.get(url)
response.raise_for_status()  # This will raise an HTTPError if the request returned an unsuccessful status code.

# Use StringIO to convert the text content into a file-like object so pd.read_csv can read it
data = StringIO(response.text)

# Read the data into a pandas DataFrame
stock_data = pd.read_csv(data)

# Initialize session state for storing orders
if 'orders' not in st.session_state:
    st.session_state.orders = []

cash = 50000000  # Example value, replace with actual cash value
# Load current portfolio
portfolio,cash,total_value = read_portfolio(cash)

# Display current portfolio
st.write("Current Portfolio")
st.dataframe(portfolio)

# Cash in the portfolio


# User input for transactions
st.write("Enter your transactions")
with st.form("order_form"):
    trans_type = st.selectbox("Transaction Type", ['Buy', 'Sell'])
    ticker = st.selectbox("ticker", stock_data['ticker'].unique())
    shares = st.number_input("Shares", min_value=1)
    submit_button = st.form_submit_button("Add Order")

    if submit_button:
        st.session_state.orders.append({'Type': trans_type, 'ticker': ticker, 'Shares': shares})

# Display and manage orders
st.write("Your Orders")
for i, order in enumerate(st.session_state.orders):
    st.write(f"{i+1}. {order['Type']} {order['Shares']} shares of {order['ticker']}")
    if st.button(f"Cancel Order {i+1}"):
        st.session_state.orders.pop(i)

# Update and display the updated portfolio
updated_portfolio,cash,total_value = update_portfolio(st.session_state.orders, portfolio.copy(),cash, stock_data)
st.write("Updated Portfolio Preview")
st.dataframe(updated_portfolio)
