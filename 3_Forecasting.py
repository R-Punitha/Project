import streamlit as st
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

@st.cache_data
def load_data():
    data = pd.read_csv("india_daily_temperature.csv")
    data['date'] = pd.to_datetime(data['date'], format="%d-%m-%Y", errors="coerce")
    data.set_index('date', inplace=True)
    return data

data = load_data()

st.header("🔮 Forecasting with ARIMA")

city_choice = st.selectbox("Select City:", data['city'].unique())
target_col = st.selectbox("Select variable to forecast:", ["temperature_2m_max", "temperature_2m_min"])

filtered = data[data['city'] == city_choice]

train_size = int(len(filtered) * 0.8)
train = filtered.iloc[:train_size]
test = filtered.iloc[train_size:]

arima_model = sm.tsa.ARIMA(train[target_col], order=(1,1,1))
arima_result = arima_model.fit()

forecast = arima_result.forecast(steps=len(test))
forecast_values = forecast.values

mae = mean_absolute_error(test[target_col], forecast_values)
rmse = mean_squared_error(test[target_col], forecast_values, squared=False)
r2 = r2_score(test[target_col], forecast_values)

st.write(f"**MAE:** {mae:.2f}, **RMSE:** {rmse:.2f}, **R²:** {r2:.2f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=test.index, y=test[target_col], mode="lines", name="Actual"))
fig.add_trace(go.Scatter(x=test.index, y=forecast_values, mode="lines", name="Forecast"))
fig.update_layout(title=f"ARIMA Forecast vs Actual ({target_col}) - {city_choice}")
st.plotly_chart(fig)
