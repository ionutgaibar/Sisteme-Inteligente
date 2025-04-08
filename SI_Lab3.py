import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

# Modelul
inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# Datele
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

plt.imshow(x_train[np.random.randint(len(x_train))])

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Antrenare
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=20,
    validation_data=(x_val, y_val),
)

# Evaluare
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# Predicții
print("Generate predictions for 3 samples")
predictions = model.predict(x_test)
print("predictions shape:", predictions.shape)

y_pred = np.argmax(predictions, axis=1)

# -------------------------------
# Evaluare suplimentară (custom)
# -------------------------------

def accuracy_score(labels, predicted_labels):
    if not len(predicted_labels) or not len(labels):
        raise ValueError('Lists must not be empty!')
    if len(predicted_labels) != len(labels):
        raise ValueError('Lists must have the same length!')
    accuracy = sum(1 for i in range(len(predicted_labels)) if predicted_labels[i] == labels[i]) / len(predicted_labels)
    return accuracy

def precision_score(labels, predicted_labels):
    if not len(predicted_labels) or not len(labels):
        raise ValueError('Lists must not be empty!')
    if len(predicted_labels) != len(labels):
        raise ValueError('Lists must have the same length!')

    tp = sum(1 for i in range(len(predicted_labels)) if predicted_labels[i] == 1 and labels[i] == 1)
    fp = sum(1 for i in range(len(predicted_labels)) if predicted_labels[i] == 1 and labels[i] == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    return precision

def macro_precision_score(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    precisions = []

    for c in classes:
        tp = sum(1 for i in range(len(y_pred)) if y_pred[i] == c and y_true[i] == c)
        fp = sum(1 for i in range(len(y_pred)) if y_pred[i] == c and y_true[i] != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precisions.append(precision)

    macro_precision = sum(precisions) / len(precisions)
    return macro_precision


# Calcul acuratețe generală (multi-class)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy (multi-class): %.2f' % accuracy)

# Transformare într-o problemă binară pentru cifra 5
target_digit = 5
y_true_binary = [1 if label == target_digit else 0 for label in y_test]
y_pred_binary = [1 if pred == target_digit else 0 for pred in y_pred]

precision = precision_score(y_true_binary, y_pred_binary)
print(f'Precision: {precision:.2f} — for digit {target_digit}')

macro_prec = macro_precision_score(y_test.astype(int), y_pred.astype(int))
print(f"Macro Precision: {macro_prec:.2f}")