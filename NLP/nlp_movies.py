def create_dataseturi():
    from tensorflow import keras

    dir_recenzii = './aclImdb'

    batch_size = 32

    train_ds = keras.utils.text_dataset_from_directory(
        f"{dir_recenzii}/train", batch_size=batch_size
    )
    val_ds = keras.utils.text_dataset_from_directory(
        f"{dir_recenzii}/val", batch_size=batch_size
    )
    test_ds = keras.utils.text_dataset_from_directory(
        f"{dir_recenzii}/test", batch_size=batch_size
    )

    return train_ds, val_ds, test_ds


train_ds, val_ds, test_ds = create_dataseturi()

for text_batch, label_batch in train_ds.take(1):
    print('Text batch:', text_batch.numpy())
    print('Label batch:', label_batch.numpy())


def vectorizare():
    from keras.src.layers import TextVectorization

    text_vectorization = TextVectorization(max_tokens=20000, output_mode="multi_hot")
    text_only_train_ds = train_ds.map(lambda x, y: x)
    text_vectorization.adapt(text_only_train_ds)
    binary_1gram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y))
    binary_1gram_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=8)
    binary_1gram_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=8)

    for text_batch, label_batch in binary_1gram_train_ds.take(1):
        print("Vectorizat:", text_batch.numpy())
        print("Etichete:", label_batch.numpy())
        for i in range(5):
            print(f"Vectorizat ({i}):", text_batch.numpy()[i][:1000])
            print(f"Etichetă ({i}):", label_batch.numpy()[i])
    print('-----------------------------------------')
    vocab = text_vectorization.get_vocabulary()
    print('-----------------------------------------')
    print("Vocabular:", vocab)
    print('-----------------------------------------')


def creare_model(max_tokens=20000, hidden_dim=16):
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(max_tokens,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optmizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def antrenare_model():
    from tensorflow import keras
    global model
    model.summary()

    history = model.fit(binary_1gram_train_ds.cache(), validation_data=binary_1gram_val_ds.cache(), epochs=10)

    print(f"Test acc: {model.evaluate(binary_1gram_test_ds)[1]:.3f}")

    def grafice(history):
        import matplotlib.pyplot as plt
        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label="Training Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, acc, label="Training Accuracy")
        plt.plot(epochs, val_acc, label="Validation Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()
    grafice(history)


def testeaza_recenzie(recenzie, afis_vect=True):
    recenzie_vectorizata = text_vectorization([recenzie])

    if afis_vect:
        print("\nRecenzie:", recenzie)
        print("Recenzie vectorizată:", recenzie_vectorizata.numpy()[0][:1000])

    probabilitate = model.predict(recenzie_vectorizata)[0][0]

    if probabilitate >= 0.5:
        sentiment = "pozitiv"
    else:
        sentiment = "negativ"

    print(f"Recenzie: {recenzie}")
    print(f"Probabilitate: {probabilitate:.3f}")
    print(f"Sentiment: {sentiment}")
