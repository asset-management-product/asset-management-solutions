# %%
import pandas as pd
from datetime import datetime

# %%


# %%
company_industry_df=pd.read_excel('E:\Downloads\master-main\master-main\company_industry.xlsx')
company_industry_df

# %%
import streamlit as st
import pandas as pd
import numpy as np
import json

import requests
from io import StringIO

income_url = 'https://raw.githubusercontent.com/penthousecompany/master/main/raw/financial_report_incomestatement_yearly_final/financial_report_incomestatement_yearly_final.csv'

def get_data_csv(url):
    response=requests.get(url)
    response.raise_for_status()
    data_raw = StringIO(response.text)
    data = pd.read_csv(data_raw)
    return data

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

income_data = get_data_csv(income_url)


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

income_data_dict = organize_income_data(income_data)
ticker = 'VNM'
year = '2019'



# %% [markdown]
# # Income Statement Structure

# %%
income_data_dict['VNM']['Appendix']

# %%
import ast

# Use a dictionary to map the stringified appendix to a list of tickers
appendix_to_tickers = {}
appendix_to_tickers[""] = []

for ticker, data in income_data_dict.items():
    # Adjusted to check for 'appendix' key in a case-insensitive manner
    appendix_data = data.get('Appendix') or data.get('appendix')
    if appendix_data:
        appendix_key = str(appendix_data)
    else:
        appendix_key = ""
    if appendix_key not in appendix_to_tickers:
        appendix_to_tickers[appendix_key] = [ticker]
    else:
        appendix_to_tickers[appendix_key].append(ticker)

# Convert the dictionary to a list of tuples for easy viewing
appendix_list = []
for appendix, tickers in appendix_to_tickers.items():
    try:
        # Safely evaluate the appendix string to a dictionary
        appendix_dict = ast.literal_eval(appendix) if appendix else {}
        appendix_list.append((appendix_dict, tickers))
    except (ValueError, SyntaxError):
        # Handle the case where the string cannot be evaluated to a dictionary
        print(f"Could not evaluate appendix: {appendix}")
        appendix_list.append(({}, tickers))

# For demonstration, let's print the appendix_list
i = 0
for appendix, tickers in appendix_list:
    i += 1
    print(f"{i}, Appendix: {appendix}, Tickers: {tickers}")


# %%
# Initialize an empty list to hold the data
data_list = []

for appendix, tickers in appendix_list:
    # Convert appendix dictionary to a string as a unique identifier
    appendix_str = str(appendix) if appendix else "No Appendix"
    # Add a dictionary for each row of data
    data_list.append({"Appendix": appendix_str, "Tickers": tickers})

# Create the DataFrame directly from the list of dictionaries
df_appendix = pd.DataFrame(data_list)

# Show the DataFrame
df_appendix

# %%
# Step 1: Collect all unique financial metrics
financial_metrics = set()
for appendix, _ in appendix_list:
    financial_metrics.update(appendix.keys())

# Ensure the financial metrics are sorted by their key to maintain order
financial_metrics = sorted(financial_metrics, key=int)

# Step 2: Create the DataFrame with financial metrics as the row index
# The financial metrics descriptions will be filled later
df = pd.DataFrame(index=financial_metrics)

# Step 3: Populate the DataFrame with financial metric descriptions for each appendix type
for i, (appendix, tickers) in enumerate(appendix_list, start=1):
    # Column name based on the appendix number (i.e., Appendix 1, Appendix 2, ...)
    column_name = f"Appendix {i}"
    # Prepare the data for this column
    column_data = {metric: appendix.get(metric, '') for metric in financial_metrics}
    # Add the column to the DataFrame
    df[column_name] = pd.Series(column_data)

# Optionally, rename the index for clarity
df.index = [f"Metric {idx}" for idx in df.index]

df.to_excel('Income Statement Structure.xlsx')

# %% [markdown]
# # Balance Sheet

# %%
bs_url = r'E:\OneDrive\6. Business Ideas\Stock Recommendation Project (SRP)\2. Database\data\raw\financial_report_balancesheet_quarterly_final\financial_report_balancesheet_quarterly_final.csv'

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
        # Check if 'Appendix' exists
        if 'Appendix' in response:
            # Append the appendix to the ticker's data
            bs_data_dict[ticker]['Appendix'] = response['Appendix']

            # Iterate over the years and metrics in the response, excluding the Appendix
            for key, metrics in response.items():
                if key != "Appendix":
                    if key not in bs_data_dict[ticker]:
                        bs_data_dict[ticker][key] = {}
                    for metric_index, value in metrics.items():
                        # Use the appendix to get the metric name, if possible
                        metric_name = response['Appendix'].get(metric_index, f"Unknown Metric {metric_index}")
                        bs_data_dict[ticker][key][metric_name] = value
        else:
            print(f"No 'Appendix' found for ticker {ticker} at row {index}")
            # Handle the case where there's no Appendix separately if needed
            # For example, you might want to simply copy the data as-is
            for key, metrics in response.items():
                bs_data_dict[key][year] = metrics

    return bs_data_dict
bs_data=pd.read_csv(bs_url)
bs_data_dict = organize_bs_data(bs_data)

# %%
import ast

# Use a dictionary to map the stringified appendix to a list of tickers
appendix_to_tickers = {}
appendix_to_tickers[""] = []

for ticker, data in bs_data_dict.items():
    # Adjusted to check for 'appendix' key in a case-insensitive manner
    appendix_data = data.get('Appendix') or data.get('appendix')
    if appendix_data:
        appendix_key = str(appendix_data)
    else:
        appendix_key = ""
    if appendix_key not in appendix_to_tickers:
        appendix_to_tickers[appendix_key] = [ticker]
    else:
        appendix_to_tickers[appendix_key].append(ticker)

# Convert the dictionary to a list of tuples for easy viewing
appendix_list = []
for appendix, tickers in appendix_to_tickers.items():
    try:
        # Safely evaluate the appendix string to a dictionary
        appendix_dict = ast.literal_eval(appendix) if appendix else {}
        appendix_list.append((appendix_dict, tickers))
    except (ValueError, SyntaxError):
        # Handle the case where the string cannot be evaluated to a dictionary
        print(f"Could not evaluate appendix: {appendix}")
        appendix_list.append(({}, tickers))

# For demonstration, let's print the appendix_list
i = 0
for appendix, tickers in appendix_list:
    i += 1
    print(f"{i}, Appendix: {appendix}, Tickers: {tickers}")

# %%
bs_data_dict['VNM']['Quý 1- 2005']

# %%
# Initialize an empty list to hold the data
data_list = []

for appendix, tickers in appendix_list:
    # Convert appendix dictionary to a string as a unique identifier
    appendix_str = str(appendix) if appendix else "No Appendix"
    # Add a dictionary for each row of data
    data_list.append({"Appendix": appendix_str, "Tickers": tickers})

# Create the DataFrame directly from the list of dictionaries
df_appendix = pd.DataFrame(data_list)

# Step 1: Collect all unique financial metrics
financial_metrics = set()
for appendix, _ in appendix_list:
    financial_metrics.update(appendix.keys())

# Ensure the financial metrics are sorted by their key to maintain order
financial_metrics = sorted(financial_metrics, key=int)

# Step 2: Create the DataFrame with financial metrics as the row index
# The financial metrics descriptions will be filled later
df = pd.DataFrame(index=financial_metrics)

# Step 3: Populate the DataFrame with financial metric descriptions for each appendix type
for i, (appendix, tickers) in enumerate(appendix_list, start=1):
    # Column name based on the appendix number (i.e., Appendix 1, Appendix 2, ...)
    column_name = f"Appendix {i}"
    # Prepare the data for this column
    column_data = {metric: appendix.get(metric, '') for metric in financial_metrics}
    # Add the column to the DataFrame
    df[column_name] = pd.Series(column_data)

# Optionally, rename the index for clarity
df.index = [f"Metric {idx}" for idx in df.index]

df.to_excel('Balance Sheet Structure.xlsx')

# %%
structure_list={}
for ticker in company_industry_df['Stock']:
    if ticker in bs_data_dict.keys():
        if 'Appendix' in bs_data_dict[ticker]:
            balance_sheet_structure = bs_data_dict[ticker]['Appendix']
        else:
            balance_sheet_structure = ""
    else:
        balance_sheet_structure = ""
    if ticker in income_data_dict.keys():
        if  'Appendix' in income_data_dict[ticker]:
            income_structure = income_data_dict[ticker]['Appendix']
        else:
            income_structure = ""
    else:
        income_structure = "" 
    structure = str(balance_sheet_structure) + str(income_structure)
    if structure not in structure_list:
        structure_list[structure]=[ticker]
    else:
        structure_list[structure].append(ticker)

i=1
for structures,tickers in structure_list.items():
    Blank = 0
    if balance_sheet_structure == "":
        Blank +=1
    
    if income_structure == "":
        Blank +=1
    print(f"{i}, Blank {Blank}",tickers,structures)
    i+=1

# %%
direct_cash_flow_url = r'E:\Downloads\master-main\master-main\raw\financial_report_cashflow_direct_yearly_final\financial_report_cashflow_direct_yearly_final.csv'
direct_cash_flow_data=pd.read_csv(direct_cash_flow_url)
direct_cash_flow_data

# %%

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
        # Check if 'Appendix' exists
        if 'Appendix' in response:
            # Append the appendix to the ticker's data
            bs_data_dict[ticker]['Appendix'] = response['Appendix']

            # Iterate over the years and metrics in the response, excluding the Appendix
            for key, metrics in response.items():
                if key != "Appendix":
                    if key not in bs_data_dict[ticker]:
                        bs_data_dict[ticker][key] = {}
                    for metric_index, value in metrics.items():
                        # Use the appendix to get the metric name, if possible
                        metric_name = response['Appendix'].get(metric_index, f"Unknown Metric {metric_index}")
                        bs_data_dict[ticker][key][metric_name] = value
        else:
            print(f"No 'Appendix' found for ticker {ticker} at row {index}")
            # Handle the case where there's no Appendix separately if needed
            # For example, you might want to simply copy the data as-is
            for key, metrics in response.items():
                bs_data_dict[key][year] = metrics

    return bs_data_dict

direct_cash_flow_data_dict = organize_bs_data(direct_cash_flow_data)

# %%
indirect_cash_flow_url = r'E:\Downloads\master-main\master-main\raw\financial_report_cashflow_indirect_yearly_final\financial_report_cashflow_indirect_yearly_final.csv'
indirect_cash_flow_data=pd.read_csv(indirect_cash_flow_url)
indirect_cash_flow_data_dict= organize_bs_data(indirect_cash_flow_data)

# %%
direct_cash_flow_data_dict['GVT']

# %%
direct_cash_flow_data

# %%
income_structure=pd.read_excel('E:\Downloads\master-main\Income Statement Structure_Official.xlsx',sheet_name='Standard')
balance_sheet_structure=pd.read_excel('E:\Downloads\master-main\Balance Sheet Structure_Official.xlsx',sheet_name='Standard')

# %%
def period_to_datetime(period):
    try:
        # For "YYYY" format
        return datetime(year=int(period.strip()), month=1, day=1)
    except ValueError:
        try:
            # For "Quý X- YYYY" format
            quarter, year = period.split('-')
            quarter_number = int(quarter.split(' ')[1])
            start_month = (quarter_number - 1) * 3 + 1
            return datetime(year=int(year.strip()), month=start_month, day=1)
        except ValueError:
            # If the period doesn't conform to expected formats, return None
            return None
        
def extract_latest_data(data_dict):
    latest_data = {}

    for ticker, periods in data_dict.items():
        valid_periods = {period: datetime for period, datetime in
                         ((period, period_to_datetime(period)) for period in periods.keys())
                         if datetime is not None}
        if valid_periods:  # Check if there are any valid periods
            latest_period = max(valid_periods, key=valid_periods.get)
            latest_data[ticker] = periods[latest_period]

    return latest_data
# Using the sample data structure to extract the latest data
latest_bs_data = extract_latest_data(bs_data_dict)
latest_income_data = extract_latest_data(income_data_dict)

# %%
latest_income_data['DGW']['18. Lợi nhuận sau thuế thu nhập doanh nghiệp(60=50-51-52)']='683,783,287,082'

# %%
# Assuming the structure has one column for the standard name and another for variants
# We'll transform this structure into a mapping dictionary
new_mapping_dict = {}

# Iterate through each row, treating the first column as the standard name
# and all other columns as containing variants.
for index, row in income_structure.iterrows():
    standard_name = row.iloc[0]  # The standard name is in the first column
    
    # Iterate through all other columns in this row to find variant names
    for variant in row[1:]:  # Skip the first column which is the standard name
        if pd.notnull(variant):  # Check if the cell is not empty
            new_mapping_dict[variant.strip()] = standard_name

for index, row in balance_sheet_structure.iterrows():
    standard_name = row.iloc[0]  # The standard name is in the first column
    
    # Iterate through all other columns in this row to find variant names
    for variant in row[1:]:  # Skip the first column which is the standard name
        if pd.notnull(variant):  # Check if the cell is not empty
            new_mapping_dict[variant.strip()] = standard_name

# %%
mapped_income_data_df = pd.DataFrame()

# %%
def map_multiple_financial_metrics_to_df(data, mapping_dict, metric_names, df=None):
    """
    Map multiple financial metric values to a DataFrame based on given metric names.
    
    Parameters:
    - data: dict, latest financial data for each ticker.
    - mapping_dict: dict, maps variant metric names to standard metric names.
    - metric_names: list of str, the standard names of the metrics to map.
    - df: pd.DataFrame, the DataFrame to update. If None, a new DataFrame is created.
    
    Returns:
    - pd.DataFrame, updated with the new financial metric columns.
    """
    if df is None:
        df = pd.DataFrame()
    
    for metric_name in metric_names:
        for ticker, metrics in data.items():
            # Initialize a placeholder for the metric value for this ticker
            metric_value = None
            
            # Check each metric to see if it's a variant of the specified metric name
            for metric, value in metrics.items():
                if metric in mapping_dict and mapping_dict[metric] == metric_name:
                    metric_value = value  # Assuming only one relevant metric per ticker
            # Add the found value to the DataFrame
            df.loc[ticker, metric_name] = metric_value
    
    return df

# Example usage:
current_asset='Current Assets and Short-term Investments'
# Define the list of metric names you want to add as columns
income_metric_names_to_add = ['Profit Before Tax', 'Profit After Tax','Gross Revenue','Current Income Tax Expense','Deferred Income Tax Expense','Interest Expense',]
bs_metric_names_to_add =['Equity','Total Assets','Inventories',current_asset,'Liabilities',"Inventories","Trade and Other Receivables","Long-term Receivables",
                         'Short-term Liabilities','Long-term Liabilities']
# Assuming latest_income_data and mock_mapping_dict are already defined
# Call the function with the list of metrics
company_finance_df = map_multiple_financial_metrics_to_df(latest_income_data, new_mapping_dict, income_metric_names_to_add)
company_finance_df = map_multiple_financial_metrics_to_df(latest_bs_data, new_mapping_dict, bs_metric_names_to_add,company_finance_df)

company_finance_df = company_finance_df.replace('', np.nan)
company_finance_df = company_finance_df.replace(',', '', regex=True).astype(float)

company_finance_df=company_finance_df.reset_index()
company_finance_df.rename(columns={'index':'ticker'},inplace=True)
company_finance_df

# %%
beta_df=pd.read_csv(r'E:\Downloads\master-main\beta.csv')

price_url=r'E:\Downloads\master-main\master-main\curated\curated_stock_ohlc.csv'
company_profile_url=r'E:\Downloads\master-main\master-main\curated\curated_company_profile.csv'
company_profile_data=pd.read_csv(company_profile_url).drop_duplicates()

price_data=pd.read_csv(price_url)
price_data=price_data[['time','close','volume','ticker']]

latest_price_data = price_data.sort_values(by=['ticker', 'time']).groupby('ticker').last().reset_index()[['ticker', 'close']]
if 'Close Price' in company_finance_df:
    pass
else:
    company_finance_df=company_finance_df.merge(latest_price_data,how='left',on='ticker')
    company_finance_df.rename(columns={'close':"Close Price"},inplace=True)

if 'L1N'in company_finance_df:
    pass
else:
    company_finance_df=company_finance_df.merge(company_industry_df[['Stock','L1N','L2N','L3N','L4N']],how='left',left_on='ticker',right_on='Stock')
    company_finance_df.drop(columns=['Stock'],inplace=True)

if "Number of outstanding shares" in company_finance_df:
    pass
else:
    company_finance_df=company_finance_df.merge(company_profile_data[['ticker','outstandingShare']],how='left',on='ticker')
    company_finance_df['outstandingShare']=company_finance_df['outstandingShare']*1000000
    company_finance_df.rename(columns={'outstandingShare':"Number of outstanding shares"},inplace=True)

if 'beta' in company_finance_df:
    pass
else:
    company_finance_df=company_finance_df.merge(beta_df,how='left',on='ticker')

# %%
risk_free_rate=7/100
income_tax=20/100

# %%
import pandas as pd


def weighted_average_roe(df, group_col, weight_col, value_col, top_n=5):
  """
  Calculates weighted average ROE within groups, considering null values.

  Args:
      df (pandas.DataFrame): The dataframe containing the data.
      group_col (str): The column name to group by (e.g., 'L1N').
      weight_col (str): The column name for weights (e.g., 'Gross Revenue').
      value_col (str): The column name for values (e.g., 'ROE').
      top_n (int, optional): The number of top companies to consider. Defaults to 5.

  Returns:
      pandas.Series: A Series containing the weighted average for each group.
  """
  def g(df):
    # Sort by group and weight column (descending) in case of nulls
    df=pd.concat([df[group_col],df[weight_col],df[value_col]],axis=1)
    df = df.sort_values(by=[group_col, weight_col], ascending=False)
    # Select top_n rows (considering nulls in weight_col)
    df = df.dropna(subset=[weight_col]).head(top_n)
    # Weighted average using Series methods
    weights = df[weight_col]
    values = df[value_col]
    return (weights * values).sum() / weights.sum()  # Avoid Series.weighted
  return df.groupby(group_col).apply(g)


# Calculate weighted average ROE for required return


# %%
company_finance_df['Tax Expense']=company_finance_df['Current Income Tax Expense']+company_finance_df['Deferred Income Tax Expense']

company_finance_df['EBIT']=company_finance_df['Profit Before Tax'] + company_finance_df['Interest Expense']


company_finance_df['ROE']=company_finance_df['Profit After Tax']/company_finance_df['Equity']
company_finance_df['ROA']=company_finance_df['Profit After Tax']/company_finance_df['Total Assets']

company_finance_df['Receivable Turnover']=company_finance_df['Gross Revenue']/company_finance_df['Trade and Other Receivables']
company_finance_df['Inventory Turnover']=company_finance_df['Gross Revenue']/company_finance_df['Inventories']
company_finance_df['CA Turnover']=company_finance_df['Gross Revenue']/company_finance_df[current_asset]
company_finance_df['TA Turnover']=company_finance_df['Gross Revenue']/company_finance_df['Total Assets']


company_finance_df['Current Ratio']=company_finance_df[current_asset]/company_finance_df['Short-term Liabilities']
company_finance_df['Interest Coverage Ratio']=company_finance_df['EBIT']/company_finance_df['Interest Expense']

company_finance_df['Leverage']=company_finance_df['Liabilities']/company_finance_df['Equity']
company_finance_df['Short-term/Total Liabilities']=company_finance_df['Short-term Liabilities']/company_finance_df['Liabilities']

company_finance_df['P/S']=company_finance_df['Close Price']/(company_finance_df['Gross Revenue']/company_finance_df['Number of outstanding shares'])
company_finance_df['P/B']=company_finance_df['Close Price']/(company_finance_df['Equity']/company_finance_df['Number of outstanding shares'])

#Industry
# Rank by Gross Revenue within L1, L2, L3, and L4 categories (assuming L columns are not null)
def rank_within_group(df, group_cols, rank_col):
    """
    Ranks data within groups defined by multiple columns.

    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        group_cols (list): A list of column names to group by.
        rank_col (str): The column name to rank by.

    Returns:
        None (Modifies the input dataframe in-place)
    """
    # Sort by group columns and ranking column
    df.sort_values(by=group_cols + [rank_col], ascending=True, inplace=True)

    # Assign ranks within each group
    for col in group_cols:
        df['Rank ' + col] = df.groupby(col)[rank_col].transform('rank',ascending=False)


rank_within_group(company_finance_df, ['L1N', 'L2N', 'L3N', 'L4N'], 'Gross Revenue')

#CAPM
company_finance_df['Required rate of return'] = company_finance_df['L1N'].map(weighted_average_roe(company_finance_df.copy(), 'L1N', 'Gross Revenue', 'ROE', 5))

company_finance_df['Discount Rate'] = np.maximum(
                                        risk_free_rate,
                                        risk_free_rate + company_finance_df['beta'] * (company_finance_df['Required rate of return'] - risk_free_rate))

company_finance_df['Discount Time Window (year)']=np.log(2)/np.log(1+company_finance_df['Discount Rate'])

company_finance_df['SaleGrowth (TTM)']=5/100
company_finance_df['SalesGrowth_03Yr']=5/100
company_finance_df['Projected SalesGrowth']=5/100
company_finance_df['PreTaxMargin (TTM)']=10/100
company_finance_df['Projected_ProfitPreTax']=company_finance_df['Gross Revenue']*(1+company_finance_df['Projected SalesGrowth'])*np.maximum(0,company_finance_df['PreTaxMargin (TTM)'])
company_finance_df['Projected_ProfitAfterTax']=company_finance_df['Projected_ProfitPreTax']*(1-income_tax) 
company_finance_df['CAPM']=company_finance_df['Projected_ProfitAfterTax']/(company_finance_df['Discount Rate'])/(1+company_finance_df['Discount Rate'])**company_finance_df['Discount Time Window (year)']

company_finance_df['Asset Quality Discount']=0 #cần tình
company_finance_df['Discounted Amount']=(company_finance_df['Inventories']+company_finance_df['Trade and Other Receivables'])*company_finance_df['Asset Quality Discount']
company_finance_df['Corrected Equity (BV)']=company_finance_df['Equity']-company_finance_df['Discounted Amount'] 
company_finance_df['Total_PresentValues']=company_finance_df['CAPM']+company_finance_df['Corrected Equity (BV)']
company_finance_df['Fair Price']=company_finance_df['Total_PresentValues']/company_finance_df['Number of outstanding shares']
company_finance_df['Opportunity']=company_finance_df['Close Price']/company_finance_df['Fair Price']-1

company_finance_df

# %%
company_finance_df.columns

# %%
import pandas as pd

def calculate_weighted_averages(df, level):
    """
    Calculates weighted averages of specified metrics by Gross Revenue for top 5 companies in each industry level.

    Args:
    - df: DataFrame containing company financials and rankings.
    - level: Industry level ('L1N', 'L2N', 'L3N', 'L4N').

    Returns:
    - DataFrame with industry name, level, and calculated weighted averages of metrics.
    """
    # Define the metrics for which to calculate weighted averages
    metrics = ['ROE', 'ROA', 'Receivable Turnover', 'Inventory Turnover', 'CA Turnover',
               'TA Turnover', 'Current Ratio', 'Interest Coverage Ratio', 'Leverage',
               'Short-term/Total Liabilities', 'P/S', 'P/B']
    
    # Filter top 5 companies based on rank within each industry level
    top_5_df = df[df[f'Rank {level}'] <= 5]
    
    # Calculate weighted averages
    weighted_avgs = top_5_df.groupby(level).apply(
        lambda x: pd.Series(
            {metric: np.average(x[metric], weights=x['Gross Revenue']) for metric in metrics}
        )
    ).reset_index()
    
    # Add industry level to the DataFrame
    weighted_avgs['Industry Level'] = level
    
    # Rename columns if necessary and reorder
    weighted_avgs = weighted_avgs.rename(columns={level: 'Industry Name'})
    weighted_avgs = weighted_avgs[['Industry Name', 'Industry Level'] + metrics]
    
    return weighted_avgs

# Example of usage for one level, you can concatenate the results for all levels
# industry_summary_df = calculate_weighted_averages(company_finance_df, 'L1N')
# For all levels, you might concatenate like this:
all_levels = ['L1N', 'L2N', 'L3N', 'L4N']
industry_summary_df = pd.concat([calculate_weighted_averages(company_finance_df, level) for level in all_levels])


# %%
# Placeholder list to collect DataFrames for each level
level_dfs = []

# Function to process each level
def process_level(df, level):
    # Get unique industries for the level
    industries = df[level].unique()
    
    # Placeholder list for processed industries
    industry_dfs = []
    
    # Process each industry
    for industry in industries:
        # Filter the DataFrame for the industry and get the top 5 companies
        industry_df = df[df[level] == industry].nsmallest(5, f'Rank {level}')
        
        # Get the list of top 5 company tickers
        top_companies = industry_df['ticker'].tolist()
        
        # Get the metrics for the top 5 companies
        metrics = industry_df[['ROE', 'ROA', 'Receivable Turnover', 'Inventory Turnover', 
                               'CA Turnover', 'TA Turnover', 'Current Ratio', 
                               'Interest Coverage Ratio', 'Leverage', 'Short-term/Total Liabilities', 
                               'P/S', 'P/B']]
        
        # Create a new DataFrame for the industry with the required structure
        data = {
            'Industry Name': industry,
            'Level': level[-2],  # Extract the numeric part of L1N, L2N, etc.
            'Top 5 Companies': ", ".join(top_companies)
        }
        
        # Add the metrics to the data dictionary
        for metric in metrics:
            data[metric] = industry_df[metric].mean()  # Calculate summarized metric, here we take the mean as example
        
        # Convert the data dictionary to a DataFrame
        industry_summary_df = pd.DataFrame([data])
        
        # Append to the list of processed industries
        industry_dfs.append(industry_summary_df)
    
    # Concatenate all industries for the level
    return pd.concat(industry_dfs, ignore_index=True)

# Process each level and concatenate results
for level in ['L4N', 'L3N', 'L2N', 'L1N']:  # Start from L4N down to L1N
    level_df = process_level(company_finance_df, level)
    level_dfs.append(level_df)

# Concatenate all levels into a final DataFrame
final_summary_df = pd.concat(level_dfs, ignore_index=True)
final_summary_df

# %%
level = ['1']  # Assuming level is a list containing industry levels
industry_name = 'Hàng tiêu dùng'
rank_companies = [1, 2, 3, 4, 5]
top_companies = ['MSN', 'VNM', 'SAB', 'PNJ', 'MCH']

# Create MultiIndex with rank as part of the index
multi_index = pd.MultiIndex.from_product(
    [level, [industry_name], rank_companies, top_companies + ['Industry Average']],
    names=['Level', 'Industry', 'Rank', 'Company'])
multi_index

# %%
def build_company_metrics_dataframe(ticker, company_finance_df, final_summary_df):
    # Define the metrics columns based on your actual data
    metrics_columns = ['ROE', 'ROA', 'Receivable Turnover', 'Inventory Turnover', 
                               'CA Turnover', 'TA Turnover', 'Current Ratio', 
                               'Interest Coverage Ratio', 'Leverage', 'Short-term/Total Liabilities', 
                               'P/S', 'P/B']  # Replace with the actual metric column names
    final_dataframe_list = []

    # Retrieve the row for the input ticker to get industry levels and names
    company_row = company_finance_df.loc[company_finance_df['ticker'] == ticker]

    # Ensure that the company exists in the DataFrame
    if company_row.empty:
        print(f"No data found for ticker {ticker}.")
        return None

    # The industry information for the ticker
    industry_info = company_row[['L1N', 'L2N', 'L3N', 'L4N']].iloc[0]

    # Iterate over the Series of levels (L1N to L4N)
    for level, industry_name in industry_info.dropna().items():
        # Get the top companies' names for the industry
        industry_row = final_summary_df.loc[final_summary_df['Industry Name'] == industry_name].iloc[0]
        top_companies = industry_row['Top 5 Companies'].split(", ")

        rank_companies = [i + 1 for i in range(len(top_companies))]
        print([level[-2]],industry_name,rank_companies,top_companies)
        # Create a MultiIndex for the top companies and industry average
        multi_index = pd.MultiIndex.from_product(
            [[level[-2]], [industry_name], ['Industry Average']+ top_companies ],
            names=['Level', 'Industry', 'Company']
        )
        
        # Retrieve the metrics for the top companies and industry average
        top_companies_metrics = company_finance_df.loc[company_finance_df['ticker'].isin(top_companies), metrics_columns]
        industry_average_metrics = final_summary_df.loc[final_summary_df['Industry Name'] == industry_name, metrics_columns].mean()
        top_companies_metrics = top_companies_metrics.round(2)  # Round to 2 decimal places
        industry_average_metrics = industry_average_metrics.round(2)  # Round and reshape


        # Combine the metrics into a single DataFrame with the MultiIndex
        combined_metrics_df = pd.concat([industry_average_metrics.to_frame().T,top_companies_metrics])
        combined_metrics_df.index = multi_index
        
        # Append to the list
        final_dataframe_list.append(combined_metrics_df)

    # Concatenate all the DataFrames into one
    final_df = pd.concat(final_dataframe_list)
    
    return final_df

# Usage example:
ticker_data_df = build_company_metrics_dataframe('VNM', company_finance_df, final_summary_df)

import streamlit as st
st.dataframe(ticker_data_df)

# %%
income_structure

# %%



