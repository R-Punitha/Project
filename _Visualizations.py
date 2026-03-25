import streamlit as st
import plotly.express as px
import pandas as pd

@st.cache_data
def load_data():
    data = pd.read_csv("india_daily_temperature.csv")
    data['date'] = pd.to_datetime(data['date'], format="%d-%m-%Y", errors="coerce")
    data.set_index('date', inplace=True)
    return data

data = load_data()

st.header("📈 Temperature Visualizations")

city_choice = st.selectbox("Select City:", data['city'].unique())

filtered = data[data['city'] == city_choice]

fig = px.line(filtered, x=filtered.index, y="temperature_2m_max", title=f"Daily Max Temperature - {city_choice}")
st.plotly_chart(fig)

fig2 = px.line(filtered, x=filtered.index, y="temperature_2m_min", title=f"Daily Min Temperature - {city_choice}")
st.plotly_chart(fig2)
