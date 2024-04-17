link='https://raw.githubusercontent.com/asset-management-product/asset-management/main'
excel='https://github.com/asset-management-product/asset-management/raw/main/Balance%20Sheet%20Structure.xlsx'

import pandas as pd
import requests
from io import StringIO
from io import BytesIO

def get_data_csv(url):
    response=requests.get(url)
    response.raise_for_status()
    data_raw = StringIO(response.text)
    data = pd.read_csv(data_raw)
    return data

def get_data_excel(url):
    """Fetches data from a URL as Excel and returns a pandas DataFrame.

    Args:
        url (str): The URL of the Excel file.

    Returns:
        pd.DataFrame: The pandas DataFrame containing the data.

    Raises:
        requests.exceptions.RequestException: If there's an error fetching the data.
    """

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for non-2xx status codes
        return pd.read_excel(BytesIO(response.content))  # Use BytesIO for binary data
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error fetching data from URL: {url}") from e

def read_github(file_name,database=None,file_path=None):
    if file_path!=None:
        file_path=file_path
        try:
            df=get_data_csv(file_path)
        except:
            df=pd.read_excel(file_path)
    else:
        if database==None:
            folder_path=''
        else:
            folder_path=f'{link}/data/{database}'
        try:
            file_path=f'{folder_path}/{file_name}.csv'
            df=get_data_csv(file_path)
        except:
            try:
                file_path=f'{folder_path}/{file_name}/{file_name}.csv'
                df=get_data_csv(file_path)
            except:
                file_path=folder_path+file_name+'xlsx'
                df=pd.read_excel(file_path)
    return df


def organize_data(data):
            data_dict = {}

            for index, row in data.iterrows():
                ticker = row['ticker']
                try:
                    response = json.loads(row['response'])
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON for ticker {ticker} at row {index}: {e}")
                    continue

                # Initialize the structure for the current ticker
                if ticker not in data_dict:
                    data_dict[ticker] = {}
                # Check if 'Appendix' exists
                if 'Appendix' in response:
                    # Append the appendix to the ticker's data
                    data_dict[ticker]['Appendix'] = response['Appendix']

                    # Iterate over the years and metrics in the response, excluding the Appendix
                    for key, metrics in response.items():
                        if key != "Appendix":
                            if key not in data_dict[ticker]:
                                data_dict[ticker][key] = {}
                            for metric_index, value in metrics.items():
                                # Use the appendix to get the metric name, if possible
                                metric_name = response['Appendix'].get(metric_index, f"Unknown Metric {metric_index}")
                                data_dict[ticker][key][metric_name] = value
                else:
                    print(f"No 'Appendix' found for ticker {ticker} at row {index}")
                    # Handle the case where there's no Appendix separately if needed
                    # For example, you might want to simply copy the data as-is
                    for key, metrics in response.items():
                        data_dict[ticker][key] = metrics

            return data_dict

