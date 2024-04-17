import streamlit as st
from functions.functions import *
df=read_github('company_overview','structured')
st.write("Fine")
st.dataframe(df)