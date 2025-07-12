# cnn_coa_tf.py

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load and preprocess data (example: MNIST instead of GWL if GWL is unavailable)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 32, 32, 1)[:, :32, :32, :]  # Adjust to 32x32 shape
x_test = x_test.reshape(-1, 32, 32, 1)[:, :32, :32, :]
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

def build_and_train_cnn(params):
    # Decode params: [num_filters, log10_learning_rate, batch_size]
    num_filters = int(params[0])
    learning_rate = 10 ** params[1]  # log-scale
    batch_size = int(params[2])

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(num_filters, (3,3), activation='relu', input_shape=(32,32,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=2,  # Use small number of epochs for fast testing
              batch_size=batch_size,
              verbose=0)

    _, acc = model.evaluate(x_test, y_test, verbose=0)
    return acc  # Fitness score

if __name__ == "__main__":
    # Example parameters: [filters, log10(learning_rate), batch_size]
    example_params = [32, -3, 64]  # 32 filters, lr=0.001, batch size 64
    score = build_and_train_cnn(example_params)
    print(f"Accuracy: {score:.4f}")
