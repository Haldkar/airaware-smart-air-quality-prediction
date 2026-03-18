# airaware-smart-air-quality-prediction
This project is an interactive Air Quality Dashboard built using Python, Streamlit, and Prophet. It allows users to:
View current AQI for a selected city

Get AQI category and health alerts

See historical trends

Predict future 7-day AQI

🚀 Features
✅ Data Preprocessing

Handles outliers using statistical method (±2 standard deviation)

Fills missing values with mean

Extracts useful date features:

Day

Month

Weekday

📊 Exploratory Data Analysis (EDA)

Daily pollutant trends

Monthly pollutant trends

Average pollutant comparison

🤖 AQI Prediction

Uses Facebook Prophet model

City-wise forecasting

Predicts next 7 days AQI

Calculates:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

🌐 Interactive Dashboard (Streamlit)

City selection dropdown

Displays:

Current AQI

AQI Category

Health Alerts

Visualizations:

Historical vs Predicted AQI

AQI Category distribution

Table view of 7-day predictions

🛠️ Tech Stack

Python 🐍

Pandas

Matplotlib

NumPy

Prophet

Streamlit
