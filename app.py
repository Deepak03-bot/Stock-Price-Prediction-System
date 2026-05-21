# Multi-Company Stock Price Prediction Dashboard (Streamlit + LSTM)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Stock Price Prediction System",
    page_icon="📈",
    layout="wide"
)

# ---------------- CUSTOM UI ----------------

st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }

    h1, h2, h3 {
        color: #00BFFF;
    }

    .stMetric {
        background-color:  grey;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333333;
    }

    .css-1d391kg {
        background-color: #111827;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------

st.sidebar.title("📊 Dashboard Menu")

st.sidebar.markdown("---")

st.sidebar.info(
    "This dashboard predicts stock prices using LSTM Deep Learning Model."
)

st.sidebar.markdown("### Features")

st.sidebar.write("✅ Multi-Company Dropdown")
st.sidebar.write("✅ Historical Analysis")
st.sidebar.write("✅ Moving Average Analysis")
st.sidebar.write("✅ LSTM Prediction")
st.sidebar.write("✅ Future 7-Day Forecast")
st.sidebar.write("✅ Candlestick chart")
st.sidebar.markdown("---")



st.title("📈 Stock Price Prediction System")

st.markdown(
    """
    <div style='background-color:#111827;padding:15px;border-radius:10px;'>
    <h3 style='color:#00BFFF;'>AI-Powered Multi-Company Stock Forecasting Dashboard</h3>
    <p style='color:white;'>Built using Machine Learning, Deep Learning (LSTM), Time Series Forecasting and Financial Analysis.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("LSTM-based Multi-Company Stock Forecasting Dashboard")

# ---------------- STOCK OPTIONS ----------------

stocks = {
    "RELIANCE": "dataset/RELIANCE.csv",
    "AXISBANK": "dataset/AXISBANK.csv",
    "COALINDIA": "dataset/COALINDIA.csv",
    "ADANIPORTS": "dataset/ADANIPORTS.csv",
    "ASIANPAINT": "dataset/ASIANPAINT.csv",
    "BAJAJ-AUTO": "dataset/BAJA-AUTO.csv",
    "BAJFINANCE": "dataset/BAJFINANCE.csv",
    "GAIL": "dataset/GAIL.csv",
    "HINDALCO": "dataset/HINDALCO.csv",
    "NTPC": "dataset/NTPC.csv",
    "ONGC": "dataset/ONGC.csv",
    "stock_metadata": "dataset/stock_metadata.csv",
    "TECHM": "dataset/TECHM.csv",
    "TITAN": "dataset/TITAN.csv",
    "ZEEL": "dataset/ZEEL.csv",
    "HCLTECH": "dataset/HCLTECH.csv",
    "INDUSINDBK": "dataset/INDUSINDBK.csv",
    "KOTAKBANK": "dataset/KOTAKBANK.csv",
    "SBIN": "dataset/SBIN.csv",
    "TATAMOTORS": "dataset/TATAMOTORS.csv",
    "TATASTEEL": "dataset/TATASTEEL.csv",
    "WIPRO": "dataset/WIPRO.csv",
    "TCS": "dataset/TCS.csv",
    "INFY": "dataset/INFY.csv",
    "HDFC": "dataset/HDFC.csv",
    "ICICIBANK": "dataset/ICICIBANK.csv"
}

selected_stock = st.selectbox(
    "Select Company",
    list(stocks.keys())
)

# ---------------- LOAD DATA ----------------

file_path = stocks[selected_stock]

df = pd.read_csv(file_path)

# Keep important columns

df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Convert Date

df['Date'] = pd.to_datetime(df['Date'])

st.subheader(f"{selected_stock} Dataset")
st.dataframe(df.head())

# ---------------- CLOSING PRICE GRAPH ----------------

st.subheader("Closing Price History")

fig1, ax1 = plt.subplots(figsize=(14,6))

ax1.plot(df['Date'], df['Close'])
ax1.set_title(f'{selected_stock} Closing Price History')
ax1.set_xlabel('Date')
ax1.set_ylabel('Closing Price')
ax1.grid(True)

st.pyplot(fig1)

# ---------------- CANDLESTICK CHART ----------------

st.subheader("Candlestick Chart")

candlestick_fig = go.Figure(
    data=[
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )
    ]
)

candlestick_fig.update_layout(
    title=f"{selected_stock} Candlestick Chart",
    xaxis_title="Date",
    yaxis_title="Price",
    height=600,
    template="plotly_dark"
)

st.plotly_chart(
    candlestick_fig,
    use_container_width=True
)

# ---------------- MOVING AVERAGES ----------------

st.subheader("Moving Average Analysis")

# Create Moving Averages

df['MA50'] = df['Close'].rolling(50).mean()
df['MA200'] = df['Close'].rolling(200).mean()

fig2, ax2 = plt.subplots(figsize=(14,6))

ax2.plot(df['Date'], df['Close'], label='Closing Price')
ax2.plot(df['Date'], df['MA50'], label='50-Day MA')
ax2.plot(df['Date'], df['MA200'], label='200-Day MA')

ax2.set_title('Moving Average Analysis')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)

# ---------------- PREPROCESSING ----------------

st.subheader("LSTM Model Training")

# Use Close Price Only

data = df[['Close']]

# Scaling

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Train-Test Split

train_size = int(len(scaled_data) * 0.8)

train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create Training Sequences

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

# Reshape

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# ---------------- BUILD MODEL ----------------

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# ---------------- TRAIN MODEL ----------------

with st.spinner("Training LSTM Model..."):
    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=0)

st.success("Model Training Completed")

# ---------------- TEST DATA ----------------

test_data_full = scaled_data[train_size - 60:]

x_test = []
y_test = scaled_data[train_size:]

for i in range(60, len(test_data_full)):
    x_test.append(test_data_full[i-60:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# ---------------- PREDICTIONS ----------------

predictions = model.predict(x_test, verbose=0)

predictions = scaler.inverse_transform(predictions)

y_test_original = scaler.inverse_transform(y_test.reshape(-1,1))

# ---------------- ACCURACY ----------------

rmse = np.sqrt(mean_squared_error(y_test_original, predictions))
mae = mean_absolute_error(y_test_original, predictions)

st.subheader("Model Accuracy")

col1, col2 = st.columns(2)

col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("MAE", f"{mae:.2f}")

# ---------------- ACTUAL VS PREDICTED GRAPH ----------------

st.subheader("Actual vs Predicted Prices")

fig3, ax3 = plt.subplots(figsize=(14,6))

ax3.plot(y_test_original, label='Actual Price')
ax3.plot(predictions, label='Predicted Price')

ax3.set_title('Actual vs Predicted Stock Prices')
ax3.set_xlabel('Days')
ax3.set_ylabel('Stock Price')
ax3.legend()
ax3.grid(True)

st.pyplot(fig3)

# ---------------- FUTURE 7 DAYS FORECAST ----------------

st.subheader("Next 7 Days Forecast")

last_60_days = scaled_data[-60:]

future_input = last_60_days.reshape(1, 60, 1)

future_predictions = []

for i in range(7):

    next_day = model.predict(future_input, verbose=0)

    future_predictions.append(next_day[0,0])

    future_input = np.append(
        future_input[:,1:,:],
        [[[next_day[0,0]]]],
        axis=1
    )

future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1,1)
)

forecast_df = pd.DataFrame({
    'Day': [f'Day {i+1}' for i in range(7)],
    'Predicted Price': [round(price[0],2) for price in future_predictions]
})

st.table(forecast_df)

# ---------------- FOOTER ----------------

st.markdown("---")
st.write("Developed using Machine Learning & Deep Learning (LSTM)")
