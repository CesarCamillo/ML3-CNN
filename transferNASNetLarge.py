import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix

import sys
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


def main (caminho):
    EPOCHS = 20
    BATCH_SIZE = 128
    NUM_CLASSES = 12
    LINE_NUM = 128
    COL_NUM = 64

    base_model = keras.applications.NASNetLarge(
        weights = 'imagenet',
        input_shape = (LINE_NUM, COL_NUM, 3),
        include_top = False
    )

    base_model.trainable = False

    inputs = keras.Input(shape = (LINE_NUM, COL_NUM, 3))
    x = base_model(inputs, training = False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs, outputs)
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam()

    model.summary()

    for inputs, targets in new_dataset:
        # Open a GradientTape.
        with tf.GradientTape() as tape:
            # Forward pass.
            predictions = model(inputs)
            # Compute the loss value for this batch.
            loss_value = loss_fn(targets, predictions)

        # Get gradients of loss wrt the *trainable* weights.
        gradients = tape.gradient(loss_value, model.trainable_weights)
        # Update the weights of the model.
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Use: transferXception.py <caminho>")

    main(sys.argv[1])