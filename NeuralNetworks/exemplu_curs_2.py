from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

test_loss, test_mae = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")

y_pred = model.predict(x_test)
y_pred_rounded = np.clip(np.rint(y_pred), 0, 9).astype(int)

print(classification_report(y_test, y_pred_rounded))

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"True: {y_test[i]}, Pred: {y_pred_rounded[i][0]}")
    plt.axis('off')
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred_rounded)

print("Confusion Matrix:")
print(conf_matrix)

false_positives, false_negatives = [0] * 10, [0] * 10
for i in range(10):
    for j in range(10):
        if i != j:
            false_negatives[i] += conf_matrix[i][j]
            false_positives[i] += conf_matrix[j][i]

for i in range(10):
    tp = conf_matrix[i][i]
    fn = false_negatives[i]
    fp = false_positives[i]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"{i}: Precision: {round(precision, 2)}, Recall: {round(recall, 2)}, F1: {round(f1, 2)}")
