
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
 
st.set_page_config(page_title="AAPL Stock Price Prediction", layout="wide")

st.title("📈 Apple Stock Price Trend Prediction (LSTM)")

df = pd.read_csv("Apple Dataset.csv", index_col='Date', parse_dates=True)
df = df[['Close']].dropna()

if st.checkbox("Show Raw Data"):
    st.write(df.tail())

df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()

def compute_RSI(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = compute_RSI(df['Close'])

st.subheader("📊 Price with SMA Indicators")
fig1, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(df['Close'], label='Close')
ax1.plot(df['SMA_50'], label='SMA 50')
ax1.plot(df['SMA_200'], label='SMA 200')
ax1.set_title("Stock Price with SMA")
ax1.legend()
st.pyplot(fig1)

st.subheader("📉 RSI (Relative Strength Index)")
fig2, ax2 = plt.subplots(figsize=(14, 4))
ax2.plot(df['RSI'], color='purple', label='RSI')
ax2.axhline(70, linestyle='--', color='red', label='Overbought')
ax2.axhline(30, linestyle='--', color='green', label='Oversold')
ax2.set_title("RSI Indicator")
ax2.legend()
st.pyplot(fig2)

data = df[['Close']].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

sequence_length = 60
X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)  
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = load_model('stock_lstm_model.h5')

predicted = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y.reshape(-1, 1)) 

st.subheader("🔮 Actual vs Predicted Price")
fig3, ax3 = plt.subplots(figsize=(14, 6))
ax3.plot(real_prices, label='Actual Price')
ax3.plot(predicted_prices, label='Predicted Price')
ax3.set_title("Actual vs Predicted Stock Price")
ax3.legend()
st.pyplot(fig3)

st.success("✅ Model successfully loaded and visualized.")
