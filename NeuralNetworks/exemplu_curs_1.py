import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Forma lui x_train: {x_train.shape}")
print(f"Forma lui y_train: {y_train.shape}")

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.show()

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = to_categorical(y_train, 10)

y_test_cat = to_categorical(y_test, 10)

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(1024, activation='tanh'), # mai bun cu relu
    Dropout(0.45),
    Dense(256, activation='leaky_relu'), # mai bun cu relu
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.15),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc="lower right")
plt.show()

plt.plot(history.history['Precision'], label='Training Precision')
plt.plot(history.history['val_Precision'], label='Validation Precision')
plt.title('Model Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend(loc="lower right")
plt.show()

plt.plot(history.history['Recall'], label='Trainig Recall')
plt.plot(history.history['val_Recall'], label='Validation Recall')
plt.title('Model Recall')
plt.xlabel('Epoch')
plt.ylabel
plt.legend(loc="lower right")
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc="upper right")
plt.show()

test_loss, test_accuracy, test_precision, test_recall = model.evaluate(x_test, y_test_cat)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")

y_pred = model.predict(x_test)
y_pred_classes = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred_classes))

# conf_matrix = confusion_matrix(y_test, y_pred_classes)

# print("Confusion Matrix:")
# print(conf_matrix)

# false_positives, false_negatives = [0] * 10, [0] * 10
# for i in range(10):
#     for j in range(10):
#         if i != j:
#             false_negatives[i] += conf_matrix[i][j]
#             false_positives[i] += conf_matrix[j][i]

# for i in range(10):
#     tp = conf_matrix[i][i]
#     fn = false_negatives[i]
#     fp = false_positives[i]
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     f1 = 2 * precision * recall / (precision + recall)
#     print(f"{i}: Precision: {round(precision, 2)}, Recall: {round(recall, 2)}, F1: {round(f1, 2)}")
