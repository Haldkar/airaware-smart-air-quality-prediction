import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("air_quality_dataset.csv")
df["Date"] = pd.to_datetime(df["Date"])

pollutants = ["pm2.5","pm10","no2","so2","co","o3"]

for col in pollutants:
    mean = df[col].mean()
    std = df[col].std()
    lower = mean - 2*std
    upper = mean + 2*std
    df[col] = df[col].where((df[col] >= lower) & (df[col] <= upper))

df[pollutants] = df[pollutants].fillna(df[pollutants].mean())

df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Weekday"] = df["Date"].dt.day_name()

print(df.describe())

df.groupby("Date")[pollutants].mean().plot()
plt.show()

df.groupby("Month")[pollutants].mean().plot()
plt.show()

df[pollutants].mean().plot(kind="bar")
plt.show()

from prophet import Prophet
import numpy as np

def forecast_aqi_city(df, city):
    city_df = df[df["City"] == city]
    df_aqi = city_df.groupby("Date")["pm2.5"].mean().reset_index()
    df_aqi.columns = ["ds", "y"]

    model = Prophet()
    model.fit(df_aqi)

    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    pred_train = model.predict(df_aqi)
    mae = np.mean(np.abs(df_aqi["y"] - pred_train["yhat"]))
    rmse = np.sqrt(np.mean((df_aqi["y"] - pred_train["yhat"])**2))

    print(city, round(mae,2), round(rmse,2))

    return forecast

def aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 150: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

def alert_level(aqi):
    if aqi > 400: return "Stay indoors"
    elif aqi > 300: return "Avoid going out"
    elif aqi > 200: return "Wear mask"
    elif aqi > 150: return "Be careful"
    else: return "Air is safe"

import streamlit as st

st.title("Air Quality Dashboard")

if "City" in df.columns:
    selected_city = st.selectbox("Select City", df["City"].unique())
    city_df = df[df["City"] == selected_city]
else:
    selected_city = "All"
    city_df = df

current = city_df[pollutants].iloc[-1].mean()

st.metric("Current AQI", int(current))
st.metric("Category", aqi_category(current))
st.metric("Alert", alert_level(current))

forecast1 = forecast_aqi_city(df, selected_city)

pred_df = forecast1[["ds","yhat"]].tail(7).copy()
pred_df.columns = ["Date","Predicted_AQI"]

pred_df["AQI_Category"] = pred_df["Predicted_AQI"].apply(aqi_category)
pred_df["Alert"] = pred_df["Predicted_AQI"].apply(alert_level)

print(pred_df)

plt.plot(pred_df["Date"], pred_df["Predicted_AQI"])
plt.xticks(rotation=45)
plt.show()

pred_df["AQI_Category"].value_counts().plot(kind="bar")
plt.show()

fig, ax = plt.subplots()
ax.plot(city_df["Date"], city_df[pollutants].max(axis=1))
ax.plot(pred_df["Date"], pred_df["Predicted_AQI"], linestyle="--")
ax.legend(["Historical AQI","Predicted AQI"])
st.pyplot(fig)

st.subheader("7-Day AQI Prediction")
st.dataframe(pred_df)

fig2, ax2 = plt.subplots()
pred_df["AQI_Category"].value_counts().plot(kind="bar", ax=ax2)
st.pyplot(fig2)

st.subheader("Historical AQI Data")
st.write(city_df)