# %% [markdown]
# ## Preluare date

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json

try:
    with open('secrets.json') as f:
        secrets = json.load(f)
except Exception as e:
    print('Error: ', e)

# %%
from binance import Client

# %%
client = Client(secrets['apiKey'], secrets['secretKey'])

# %%
historical = client.get_historical_klines('BTCEUR', Client.KLINE_INTERVAL_1HOUR, '12 Nov 2024')

# %%
historical

# %%
close_values = [kline[4] for kline in historical]
close_values

# %%
len(close_values)

# %% [markdown]
# ## Preprocesare date

# %%

data = np.array(close_values, dtype=np.float64).reshape(-1, 1)
data

# %%

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
data_scaled

# %% [markdown]
# ## Creare dataset

# %%


def create_dataset(data, look_back=10):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back, 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)


# %%
look_back = 4
X, y = create_dataset(data_scaled, look_back)

X.shape

# %%
X = X.reshape(X.shape[0], X.shape[1], 1)

X.shape

# %% [markdown]
# ## Antrenare model LSTM

# %%

# %%
train_size = int(len(X) * 0.70)
test_size = len(X) - train_size
train_size, test_size

# %%
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# %%
model = Sequential()
model.add(LSTM(200, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(200))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=1000, batch_size=64, verbose=1)

# %% [markdown]
# ## Testare model LSTM

# %%
predictions = model.predict(X_test)

# %%
predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# %%

# %%
plt.plot(y_test_rescaled, label='True')
plt.plot(predictions_rescaled, label='Predicted')
plt.legend()
plt.show()

# %%
mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
rmse = np.sqrt(mse)
print('MAE: ', mae)
print('MSE: ', mse)
print('RMSE: ', rmse)
