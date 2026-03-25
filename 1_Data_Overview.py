import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    data = pd.read_csv("india_daily_temperature.csv")
    data['date'] = pd.to_datetime(data['date'], format="%d-%m-%Y", errors="coerce")
    data.set_index('date', inplace=True)
    return data

data = load_data()

st.header("📊 Data Overview")
st.write(data.head())
st.write(data.describe())
