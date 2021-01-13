from data import Spiral
from net import SpiralNetV1
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

classes = 6
split = 0.2
model = SpiralNetV1(outputs=classes).model
ds = Spiral(100000, classes)
(X, Y) = ds.data

(x_train, y_train), (x_test, y_test) = (X[:int(split*len(X))], Y[:int(split*len(X))]), ((X[int(split*len(X)):], Y[int(split*len(X)):]))
(x_train, y_train) = (x_train[int(len(x_train)/2):], y_train[int(len(x_train)/2):])
(x_val, y_val) = (x_train[:int(len(x_train)/2)], y_train[:int(len(x_train)/2)])

model.summary()

model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=100,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)

history.history

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions")
predictions = model.predict(x_test)
print("predictions shape:", predictions.shape)

pred_labels = np.array(list(np.argmax(pred) for pred in predictions))
print(pred_labels)

ds.test()

plt.scatter(x_test[:,0], x_test[:,1], c=pred_labels)
plt.show()