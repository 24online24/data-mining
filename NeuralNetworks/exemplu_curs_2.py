import tensorflow as tf


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential([
  Flatten(input_shape=(28, 28)),
  Dense(128, activation='relu'),
  Dense(64, activation='relu'),
  Dense(1, activation='linear')
])

history = model.fit(x_train, y_train, epochs = 10, batch_size=32, validation_split=0.1)

test_lost, test_mae = model.evaluate