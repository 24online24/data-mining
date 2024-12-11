import json

import matplotlib.pyplot as plt
import numpy as np
from binance import Client
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

try:
    with open('secrets.json') as f:
        secrets = json.load(f)
except Exception as e:
    print('Error: ', e)


client = Client(secrets['apiKey'], secrets['secretKey'])

historical = client.get_historical_klines('BTCEUR', Client.KLINE_INTERVAL_1HOUR, '12 Nov 2024')

close_values = [kline[4] for kline in historical]
data = np.array(close_values, dtype=np.float64).reshape(-1, 1)


scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)


def create_dataset(data, look_back=10):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back, 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)


look_back = 8
X, y = create_dataset(data_scaled, look_back)

X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(X) * 0.70)
test_size = len(X) - train_size
train_size, test_size

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(100))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=1000, batch_size=64, verbose=1)


predictions = model.predict(X_test)

predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.plot(y_test_rescaled, label='True')
plt.plot(predictions_rescaled, label='Predicted')
plt.legend()
plt.show()

mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
rmse = np.sqrt(mse)
print('MAE: ', mae)
print('MSE: ', mse)
print('RMSE: ', rmse)

next_hour = model.predict(X[-1:])
next_hour_rescaled = scaler.inverse_transform(next_hour.reshape(-1, 1))
print('\nClose ora curentă: ', next_hour_rescaled[0][0])
