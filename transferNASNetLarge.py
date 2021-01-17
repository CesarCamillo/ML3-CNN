# -*- encoding: iso-8859-1 -*-

import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow

import sys
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

def main (caminho):
    EPOCHS = 10
    BATCH_SIZE = 32
    NUM_CLASSES = 12
    LINE_NUM = 128 # Tamanho Mínimo é 32
    COL_NUM = 128 # Tamanho Mínimo é 32

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       validation_split=0.1,
                                       featurewise_std_normalization=True)

    train_generator = train_datagen.flow_from_directory(
        directory = caminho,
        target_size=(LINE_NUM,COL_NUM),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        directory = caminho, # same directory as training data
        target_size=(LINE_NUM, COL_NUM),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    base_model = keras.applications.NASNetMobile(
        weights = 'imagenet',
        input_shape = (LINE_NUM, COL_NUM, 3),
        include_top = False
    )

    base_model.trainable = False

    inputs = keras.Input(shape = (LINE_NUM, COL_NUM, 3))
    x = base_model(inputs, training = False)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)

    outputs = keras.layers.Dense(12)(x)
    model = keras.Model(inputs, outputs)

    model.summary()

    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(train_generator, epochs = EPOCHS, validation_data = validation_generator)

    base_model.trainable = True
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(train_generator, epochs = EPOCHS, validation_data = validation_generator)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Use: transferXception.py <caminho>")

    main(sys.argv[1])