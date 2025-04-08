import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

dataset.isna().sum()

dataset = dataset.dropna()

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='', dtype=int)
dataset.tail()

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')

train_dataset.describe().transpose()

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

train_features.head()

for col in train_features.columns:
  train_features[col] = train_features[col] / train_features[col].max()
  test_features[col] = test_features[col] / test_features[col].max()

train_features.head()

def build_and_compile_model():
  inputs = keras.Input(shape=(len(train_features.columns),))

  x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
  x = layers.Dense(64, activation="relu", name="dense_2")(x)

  outputs = layers.Dense(1)(x)

  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      loss='mean_absolute_error',
      optimizer=tf.keras.optimizers.Adam(0.0001))

  return model

dnn_model = build_and_compile_model()
dnn_model.summary()

#time
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=1, epochs=100)

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)

plot_loss(history)

def build_and_compile_improved_model():
    inputs = keras.Input(shape=(len(train_features.columns),))

    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)

    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='mean_absolute_error',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    return model

improved_model = build_and_compile_improved_model()
improved_model.summary()

class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_loss') < 2.0:
            print(f"\n✔ Val loss below 2.0 at epoch {epoch}, stopping training.")
            self.model.stop_training = True

custom_stop = CustomEarlyStopping()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history_improved = improved_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=1,
    epochs=2000,
    callbacks=[custom_stop]
)

plot_loss(history_improved)

improved_predictions = improved_model.predict(test_features)
predictions = dnn_model.predict(test_features)

print(f'First three predictions are {predictions[:3].squeeze()} and first three labels are {test_labels.values[:3]}')

def mae(labels, predicted_labels):
    if not len(predicted_labels) or not len(labels):
        raise ValueError('Lists must not be empty!')
    if len(predicted_labels) != len(labels):
        raise ValueError('Lists must have the same length!')

    mae = sum(abs(predicted_labels[i] - labels[i]) for i in range(len(predicted_labels))) / len(predicted_labels)
    return mae

# Apelare funcție pe test set
y_true = test_labels.values
y_pred = predictions.squeeze()

mae_result = mae(y_true, y_pred)
print(f"Mean Absolute Error: {mae_result:.2f}")

improved_predictions = improved_model.predict(test_features)
improved_mae_result = mae(test_labels.values, improved_predictions.squeeze())
print(f"Improved Mean Absolute Error: {improved_mae_result:.2f}")

print(len(dataset))
