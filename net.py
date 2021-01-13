""" Module responsible for aggregate the distinctive net configurations used for training."""

import tensorflow
from tensorflow import keras
from tensorflow.keras import layers

class SpiralNetV1:
    def __init__(self, input=2, outputs=2):
        self._model = keras.Sequential(name="SpiralNetV1")
        self._model.add(keras.Input(shape=(input,)))
        self._model.add(layers.Dense(16, activation="relu"))
        self._model.add(layers.Dense(16, activation="relu"))
        self._model.add(layers.Dense(16, activation="relu"))
        self._model.add(layers.Dense(16, activation="relu"))
        self._model.add(layers.Dense(16, activation="relu"))
        self._model.add(layers.Dense(16, activation="relu"))
        self._model.add(layers.Dense(16, activation="relu"))
        self._model.add(layers.Dense(16, activation="relu"))
        self._model.add(layers.Dense(16, activation="relu"))
        self._model.add(layers.Dense(16, activation="relu"))
        self._model.add(layers.Dense(16, activation="relu"))
        self._model.add(layers.Dense(outputs, activation="softmax", name="output"))
    
    @property
    def model(self):
        return self._model