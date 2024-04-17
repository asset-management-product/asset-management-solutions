import streamlit as st
import pandas as pd
import numpy as np
import json

import requests
from io import StringIO
from functions.functions import *

# Create tabs
list_tabs=["Basic Info", "Financial Statements","Deal And Events"]
basic_info, finance, events = st.tabs(list_tabs)

#Data 
profile_url = ' https://raw.githubusercontent.com/penthousecompany/master/main/curated/curated_company_profile.csv'
income_url = 'https://raw.githubusercontent.com/penthousecompany/master/main/raw/financial_report_incomestatement_yearly_final/financial_report_incomestatement_yearly_final.csv'
bs_url = r'E:\OneDrive\6. Business Ideas\Stock Recommendation Project (SRP)\2. Database\data\raw\financial_report_balancesheet_quarterly_final\financial_report_balancesheet_quarterly_final.csv'
direct_cash_flow_url = r'E:\Downloads\master-main\master-main\raw\financial_report_cashflow_direct_yearly_final\financial_report_cashflow_direct_yearly_final.csv'
indirect_cash_flow_url = r'E:\Downloads\master-main\master-main\raw\financial_report_cashflow_indirect_yearly_final\financial_report_cashflow_indirect_yearly_final.csv'

def finance_dict_to_dataframe(ticker_data_dict):
    # Initialize a list to store the data and a set for unique accounts
    data_for_df = []
    ordered_accounts = []
    
    # Iterate over each year in the dictionary
    for year, accounts in ticker_data_dict.items():
        if year != "Appendix":  # Exclude the appendix
            for account, value in accounts.items():
                # Append data for DataFrame
                data_for_df.append({"Year": year, "Account": account, "Value": value})
                # Keep track of the order of accounts
                if account not in ordered_accounts:
                    ordered_accounts.append(account)
    
    # Create a DataFrame
    df = pd.DataFrame(data_for_df)
    
    # Reshape the DataFrame to have years as columns and accounts as rows
    try:
        df_pivoted = df.pivot(index="Account", columns="Year", values="Value").reset_index()
    except KeyError as e:
        print(f"KeyError occurred: {e}")
        st.write("Data doesn't exist or Error")
        # Handle the error or debug further
        return pd.DataFrame()  # Or return None, based on your use case
    # Reorder the DataFrame rows based on the original account order
    df_ordered = df_pivoted.set_index("Account").reindex(ordered_accounts).reset_index()
    
    return df_ordered

def select_ticker(ticker_list):
    user_input_ticker = st.selectbox('Select a Ticker', ticker_list)

    # Function to handle option selection
    # Process input
    if user_input_ticker:
        # Normalize input and company display names to lowercase for case-insensitive matching
        st.session_state['selected_company']=user_input_ticker
    else:
        st.write("There's no company matching")
    if st.session_state.get('selected_company'):
        selected_option = st.session_state['selected_company']
        selected_ticker = selected_option.split(' - ')[0]
    return selected_ticker



# Make a GET request to fetch the raw CSV content
# Read the data into a pandas DataFrame
company_data = get_data_csv(profile_url)
income_data = get_data_csv(income_url)
bs_data=pd.read_csv(bs_url)
direct_cash_flow_data=pd.read_csv(direct_cash_flow_url)
indirect_cash_flow_data=pd.read_csv(indirect_cash_flow_url)
company_industry_df=pd.read_excel('E:\Downloads\master-main\master-main\company_industry.xlsx')
company_data.drop_duplicates(inplace=True)

income_data_dict = organize_data(income_data)
bs_data_dict = organize_data(bs_data)
direct_cash_flow_data_dict = organize_data(direct_cash_flow_data)
indirect_cash_flow_data_dict = organize_data(indirect_cash_flow_data)



shareholder_info=company_data[['ticker','shareHolder_OwnPercent','shortName']]
shareholder_info['display']= shareholder_info['ticker'] + ' - ' + shareholder_info['shortName']
# company_data=company_data[['ticker','shortName','industry','industryEn','exchange',
#                            'noShareholders','foreignPercent','outstandingShare',
#                            'issueShare','establishedYear','noEmployees',
#                            'website','companyProfile','historyDev','companyPromise',
#                            'businessRisk','keyDevelopments','businessStrategies']]# Input widget to accept ticker or company name


# Input widget to accept part of ticker or company name
#user_input = st.text_input('Enter Company Ticker or Name:', on_change=None, key="user_input")


selectbox_container = st.container()
with selectbox_container:
    selected_ticker=select_ticker(company_industry_df['Stock'])

with basic_info:
    st.write(
        """
        Comming Soon: \n
        \t Market Capitalization (bil.) \n
        \t Free Shares (mil.) \n 
        \t Outstanding Shares (mil.) \n 
         Major Ownership 
 \b Institutional Domestic
 \n 
 Individual Domestic 
 Institutional Foreign 
 Individual Foreign 

 \b Total Ownership 
 State Ownership 
 Foreign Ownership 
 Other Ownership 

"""
    )
    # Show selected company's ownership percentage
        
        # Retrieve the ownership information string
    if st.session_state.get('selected_company'):
        try:
            ownership_info_str = company_data[company_data['ticker'] == selected_ticker]['shareHolder_OwnPercent'].values[0]
            
            # Convert the string representation of the list of dictionaries into an actual list of dictionaries
            # Note: This step assumes that ownership_info_str is a string that needs to be evaluated into a Python object.
            # If it's already a list of dictionaries, you can skip the eval step.
            ownership_info = eval(ownership_info_str)
            
            
            # Initialize lists to store shareholder names and their ownership percentages
            shareholders = []
            percentages = []
            
            # Extract shareholder names and their percentages
            for shareholder_dict in ownership_info:
                for name, percentage in shareholder_dict.items():
                    shareholders.append(name)
                    percentages.append(percentage)
            
            # Create a DataFrame from the lists
            ownership_df = pd.DataFrame({
                'Shareholder': shareholders,
                'Ownership Percentage': percentages
            })

            # Formatting the 'Ownership Percentage' column to display as percentage
            ownership_df['Ownership Percentage'] = ownership_df['Ownership Percentage'].apply(lambda x: f'{x:.2%}')

            # Display the formatted DataFrame
            st.write(f"Ownership Percentage for {selected_ticker}:")
            st.dataframe(ownership_df)
        except:
            st.write("There is no Company data about this company yet")

    # Optionally display the full company data
    if st.checkbox('Show Full Company Data'):
        st.write(company_data)
    
with finance:
    income_statement_section, balance_sheet_section,cash_flow_section = st.tabs(["Income Statement", "Balance Sheet",'Cash Flow'])

    with income_statement_section:
        st.subheader("Income Statement - Báo cáo Kết quả kinh doanh")
        # Your content for nested tab 1
        #selected_ticker=select_ticker('tab-2')
        # Convert the nested dictionary into a DataFrame
        # Function to convert the dictionary to a DataFrame while preserving the order
        st.dataframe(finance_dict_to_dataframe(income_data_dict[selected_ticker]))


    with balance_sheet_section:
        st.subheader("Balance Sheet - Bảng cân đối kế toán")
        # Your content for nested tab 2
        st.dataframe(finance_dict_to_dataframe(bs_data_dict[selected_ticker]))
    with cash_flow_section:
        direct_cash_flow_section, indirect_cash_flow_section = st.tabs(["Direct Cash Flow", "Indirect Cash Flow"])
        with direct_cash_flow_section:

            st.dataframe(finance_dict_to_dataframe(direct_cash_flow_data_dict[selected_ticker]))
        with indirect_cash_flow_section:
            
            st.write(selected_ticker)
            st.dataframe(finance_dict_to_dataframe(indirect_cash_flow_data_dict[selected_ticker]))

with events:
    st.header(list_tabs[1])
    company_insider_deal_url = 'https://raw.githubusercontent.com/penthousecompany/master/main/structured/structured_company_insider_deals.csv'
    insider_deal=get_data_csv(company_insider_deal_url)
    st.write("Insider Deal, Lưu ý, hiện tại các ticker ở Insider Deal bị lệch nhiều so với Ticker thông thường")
    st.write(insider_deal)#[insider_deal['ticker']==selected_ticker])

#ticker,dealAnnounceDate,dealMethod,dealAction,dealQuantity,dealPrice,dealRatio

    company_events_url = 'https://raw.githubusercontent.com/penthousecompany/master/main/structured/structured_company_events.csv'
    company_events_full=get_data_csv(company_events_url)
    company_events=company_events_full[['ticker','price','priceChange','eventName','eventCode','notifyDate','exerDate','regFinalDate','exRigthDate','eventDesc','eventNote']]
    st.write("Company Event")
    st.write(company_events[company_events['ticker']==selected_ticker])
    #datime,id,ticker,price,priceChange,priceChangeRatio,priceChangeRatio1W,priceChangeRatio1M,eventName,eventCode,notifyDate,exerDate,regFinalDate,exRigthDate,eventDesc,eventNote