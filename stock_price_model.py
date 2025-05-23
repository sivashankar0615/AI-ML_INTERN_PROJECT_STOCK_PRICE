import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

stock = pd.read_csv('Apple Dataset.csv', index_col='Date', parse_dates=True)
print("Sample data:\n", stock.head())

plt.figure(figsize=(14, 6))
plt.plot(stock['Close'], label='Close Price')
plt.title('AAPL Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

data = stock[['Close']].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

X = []
y = []
sequence_length = 60

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(14, 6))
plt.plot(real_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

stock['SMA_50'] = stock['Close'].rolling(window=50).mean()
stock['SMA_200'] = stock['Close'].rolling(window=200).mean()

plt.figure(figsize=(14,6))
plt.plot(stock['Close'], label='Close')
plt.plot(stock['SMA_50'], label='SMA 50')
plt.plot(stock['SMA_200'], label='SMA 200')
plt.title('Stock Price with SMA Indicators')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

model.save('stock_lstm_model.h5')
print("Model saved successfully as 'stock_lstm_model.h5'")
