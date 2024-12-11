import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from itertools import product
import pandas as pd

param_grid = {
    'look_back': [3, 6, 10],
    'lstm_units': [50, 100, 200],
    'lstm_activation': ['tanh', 'relu'],
    'lstm_recurrent_activation': ['sigmoid', 'relu'],
    'dense_activation': [None, 'relu', 'sigmoid'],
    'epochs': [10, 20, 50]
}


def build_evaluate_model(X_train, y_train, X_test, y_test, look_back, lstm_units, lstm_activation, lstm_recurrent_activation, dense_activation, epochs):
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=False, input_shape=(look_back, 1), activation=lstm_activation, recurrent_activation=lstm_recurrent_activation))
    model.add(Dense(1, activation=dense_activation))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0)

    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
    return mae, rmse


np.random.seed(42)

data = np.sin(np.linspace(0, 100, 500))

data = data.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

test_data = np.sin(np.linspace(100, 150, 200)).reshape(-1, 1)
test_data_scaled = scaler.transform(test_data)


def create_dataset(data, look_back=10):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back, 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)


results = []
i = 1
for params in product(*param_grid.values()):
    print(i)
    i += 1
    look_back, lstm_units, lstm_activation, lstm_recurrent_activation, dense_activation, epochs = params

    X, y = create_dataset(data_scaled, look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    X_test, y_test = create_dataset(test_data_scaled, look_back)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    mae, rmse = build_evaluate_model(X, y, X_test, y_test, look_back, lstm_units, lstm_activation, lstm_recurrent_activation, dense_activation, epochs)

    results.append({
        'look_back': look_back,
        'lstm_units': lstm_units,
        'lstm_activation': lstm_activation,
        'lstm_recurrent_activation': lstm_recurrent_activation,
        'dense_activation': dense_activation,
        'epochs': epochs,
        'mae': mae,
        'rmse': rmse
    })

results_df = pd.DataFrame(results)

print('\nAll Results Sorted by RMSE:')
print(results_df.sort_values('rmse').to_string(index=False))
