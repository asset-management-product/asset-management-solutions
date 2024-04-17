import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json

# Set page config to use full page width
st.set_page_config(layout="wide")
st.title('Financial Data Viewer')

# Load the data
@st.cache_data
def load_data():
    file_path = r'E:\Downloads\master-main\master-main\raw\financial_report_incomestatement_yearly_final\financial_report_incomestatement_yearly_final.csv'
    return pd.read_csv(file_path)

df = load_data()

# Function to parse and process JSON data
def process_json(data):
    try:
        json_data = json.loads(data)
        financial_metrics = json_data.get("CHỈ TIÊU", {})
        years_data = {year: {financial_metrics.get(str(index), ''): value 
                             for index, value in year_data.items() if str(index) in financial_metrics}
                      for year, year_data in json_data.items() if year != "CHỈ TIÊU"}
        return years_data
    except ValueError:
        return {}  # Return empty dict if JSON is invalid

# Function to plot the line graph
def plot_line_graph(df, primary_metrics, secondary_metrics):
    if df.empty or (not primary_metrics and not secondary_metrics):
        st.write("No data to display. Please select metrics.")
        return

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Primary Metrics', color=color)

    for metric in primary_metrics:
        ax1.plot(df.index, df[metric], label=metric, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

    if secondary_metrics:
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Secondary Metrics', color=color)
        for metric in secondary_metrics:
            ax2.plot(df.index, df[metric], label=metric, linestyle='--', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    
    # Position the primary legend
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
    
    # Position the secondary legend directly below the primary legend
    if secondary_metrics:
        ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.85), title='Secondary Metrics')

    plt.title('Financial Metrics Over Time')
    plt.legend()
    st.pyplot(fig)

# Preprocess the data
tickers_data = {}
for index, row in df.iterrows():
    ticker = row['ticker']
    tickers_data[ticker] = process_json(row['response'])

# Streamlit UI components
ticker_selected = st.selectbox('Select a Ticker', list(tickers_data.keys()))

years_available = []
financial_metrics = []
if ticker_selected:
    years_available = sorted(tickers_data[ticker_selected].keys())
    if years_available:
        financial_metrics = list(tickers_data[ticker_selected][years_available[0]].keys())

year_range = st.select_slider(
    'Select a Year Range',
    options=years_available,
    value=(min(years_available), max(years_available)) if years_available else (None, None)
)

# Collapsible section for financial metrics
with st.expander("Select Financial Metrics"):
    metrics_selected = st.multiselect('', financial_metrics, default=financial_metrics)

# Display data
if ticker_selected and metrics_selected and year_range:
    start_year, end_year = year_range
    with st.expander("View Financial Data Table"):
        filtered_data = {int(float(year)): {metric: tickers_data[ticker_selected][year].get(metric, None) 
                                for metric in metrics_selected}
                         for year in years_available if start_year <= year <= end_year}

        # Convert to DataFrame for display and set financial metrics as index
        df_display = pd.DataFrame(filtered_data)
        df_display_for_plotting = df_display.apply(pd.to_numeric, errors='coerce')  # Convert to numeric for plotting
        # Apply formatting for thousands separator
        pd.options.display.float_format = '{:,.0f}'.format
        df_display_formatted = df_display.applymap(lambda x: '{:,.0f}'.format(x) if isinstance(x, (int, float)) else x)

        # Convert DataFrame to HTML for display with custom styling
        html = df_display_formatted.to_html(classes='table table-striped', border=0, escape=False, index=True)
        html = html.replace('<table ', '<table style="white-space: nowrap;text-align:right;" ')
        st.markdown(html, unsafe_allow_html=True)

    
    # Options for line graph
    with st.expander("Select Metrics for Graph"):
        primary_metrics = st.multiselect('Select Primary Metrics for Graph', financial_metrics, key='primary')
        secondary_metrics = st.multiselect('Select Secondary Metrics for Secondary Axis', financial_metrics, key='secondary')
        
        

        if primary_metrics or secondary_metrics:
            plot_line_graph(df_display_for_plotting.transpose(), primary_metrics, secondary_metrics)

# Run the Streamlit app
if __name__ == '__main__':
    load_data()
