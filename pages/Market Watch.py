import streamlit as st
import pandas as pd
import numpy as np
import json

import requests
from io import StringIO

from functions.functions import *


st.header("Market Watch")

st.write("Currently there are three sections: Benchmark Performance, Industry View and Player View")
st.write("In Alpha Phase")

list_tabs=["Benchmark Performance", "Industry View", "Player View"]
tab0, tab1, tab2 = st.tabs(list_tabs)
st.image("https://i.imgur.com/O6919hj.jpg", caption='Benchmark Performance Requirement', use_column_width=True)

income_file_name='financial_report_incomestatement_yearly_final'
bs_file_name = 'financial_report_balancesheet_quarterly_final'

def expand_financial_response_column(row):
    # Parse the JSON data from the 'response' column
    response_data = json.loads(row['response'])

    # Extract the 'Appendix' data, which we expect to be a dictionary
    appendix_data = response_data.get('Appendix', {})

    # Now we will flatten the appendix_data if it is not already a string
    if isinstance(appendix_data, str):
        # If it's a string, it might be JSON-encoded, so we attempt to parse it
        appendix_data = json.loads(appendix_data)

    # Then we extract items from this dictionary, assuming each value is just a string
    flattened_data = {k: v for k, v in appendix_data.items()}

    return pd.Series(flattened_data)

income_data = read_github(income_file_name,'raw')
bs_data= read_github(bs_file_name)
latest_update=str(income_data['crawl_datime'][0])
# Example: If latest_update is a datetime series with one element


# Now you can slice the string safely
year = str(20)+latest_update[0:2]
month = latest_update[2:4]
day = latest_update[4:6]

st.write("Cập nhật lần cuối vào","Năm", year, "Tháng", month, "Ngày", day)


def organize_income_data(income_data):
    income_data_dict = {}
    for index, row in income_data.iterrows():
        ticker = row['ticker']
        try:
            response = json.loads(row['response'])  # Convert JSON string to dictionary
        except json.JSONDecodeError:
            print(f"Error decoding JSON for ticker {ticker} at row {index}")
            continue

        # Initialize the structure for the current ticker
        if ticker not in income_data_dict:
            income_data_dict[ticker] = {}

        # Check if 'Appendix' exists
        if 'Appendix' in response:
            # Append the appendix to the ticker's data
            income_data_dict[ticker]['Appendix'] = response['Appendix']

            # Iterate over the years and metrics in the response, excluding the Appendix
            for year, metrics in response.items():
                if year != "Appendix":
                    if year not in income_data_dict[ticker]:
                        income_data_dict[ticker][year] = {}
                    for metric_index, value in metrics.items():
                        # Use the appendix to get the metric name, if possible
                        metric_name = response['Appendix'].get(metric_index, f"Unknown Metric {metric_index}")
                        income_data_dict[ticker][year][metric_name] = value
        else:
            print(f"No 'Appendix' found for ticker {ticker} at row {index}")
            # Handle the case where there's no Appendix separately if needed
            # For example, you might want to simply copy the data as-is
            for year, metrics in response.items():
                income_data_dict[ticker][year] = metrics

    return income_data_dict

def organize_bs_data(bs_data):
    bs_data_dict = {}

    for index, row in bs_data.iterrows():
        ticker = row['ticker']
        try:
            response = json.loads(row['response'])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for ticker {ticker} at row {index}: {e}")
            continue

        # Initialize the structure for the current ticker
        if ticker not in bs_data_dict:
            bs_data_dict[ticker] = {}

        # Handling cases where "Appendix" might be missing
        appendix = response.get("Appendix", {})

        # Extract and organize financial data by year
        for key, value in response.items():
            # Skip "Appendix" key as it's used for naming metrics
            if key == "Appendix":
                continue

            # Yearly data processing
            if key not in bs_data_dict[ticker]:
                bs_data_dict[ticker][key] = {}
            for metric_index, metric_value in value.items():
                metric_name = appendix.get(str(metric_index), f"Unknown Metric {metric_index}")
                bs_data_dict[ticker][key][metric_name] = metric_value

    return bs_data_dict

income_data_dict = organize_income_data(income_data)
ticker = 'VNM'
year = '2019'
st.header(f"{ticker},{year}")
st.write(income_data_dict[ticker][year])
#st.write(bs_data['response'][5])
bs_data_dict = organize_bs_data(bs_data)

#ROE=income_data_dict[ticker][year]/