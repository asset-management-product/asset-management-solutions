import streamlit as st
import pandas as pd
import json
import requests
from io import StringIO

# Corrected decorator and added function for loading data
@st.cache(suppress_st_warning=True)
def load_data(url):
    # Make a GET request to fetch the raw CSV content
    response = requests.get(url)
    response.raise_for_status()  # This will raise an HTTPError if the request returned an unsuccessful status code.
    
    # Use StringIO to convert the text content into a file-like object so pd.read_csv can read it
    data = StringIO(response.text)
    
    # Read the data into a pandas DataFrame
    return pd.read_csv(data)

# Load the data
url = 'https://raw.githubusercontent.com/penthousecompany/master/main/raw/financial_report_incomestatement_yearly_final/financial_report_incomestatement_yearly_final.csv'
df = load_data(url)



# Function to parse JSON data
def parse_json(data):
    try:
        return json.loads(data)
    except ValueError:
        return {}  # Return empty dict if JSON is invalid

# Preprocess the data
tickers_data = {}
for index, row in df.iterrows():
    ticker = row['ticker']
    json_data = parse_json(row['response'])
    tickers_data[ticker] = json_data

# Streamlit selectors
ticker_selected = st.selectbox('Select a Ticker', list(tickers_data.keys()))

# Determine the available years from the selected ticker
years_available = set()
if ticker_selected:
    for metric in tickers_data[ticker_selected].values():
        years_available.update(metric.keys())
years_available = sorted(years_available)
# Ensure there are available years to select from
if years_available:
    # Year range selection
    start_year, end_year = st.select_slider(
        'Select a Year Range',
        options=years_available,
        value=(min(years_available), max(years_available))
    )
else:
    st.warning("No years available for the selected ticker. Please select a different ticker.")
    start_year, end_year = None, None

# Financial metrics selection
financial_metrics = list(tickers_data[ticker_selected].keys())
metrics_selected = st.multiselect('Select Financial Metrics', financial_metrics, default=financial_metrics)

# Display data
if ticker_selected and metrics_selected:
    filtered_data = {metric: {year: tickers_data[ticker_selected][metric].get(year, None) 
                              for year in years_available if start_year <= year <= end_year} 
                     for metric in metrics_selected}

    # Convert to DataFrame for display
    df_display = pd.DataFrame(filtered_data)
    st.table(df_display.transpose())
