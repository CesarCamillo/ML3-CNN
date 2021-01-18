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
    LINE_NUM = 128
    COL_NUM = 64

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

    base_model = keras.applications.EfficientNetB2(
        weights = 'imagenet',
        include_top = False,
        input_shape = (LINE_NUM, COL_NUM, 3),
    )

    base_model.trainable = False

    inputs = keras.Input(shape = (LINE_NUM, COL_NUM, 3))
    x = base_model(inputs, training = False)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    #As camas as seguir são adicionadas para simular o comportamento de SVM posteriormente
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dense(1), W_regularizer=l2(0.01))(x)
    x = keras.layers.activation('linear'))(x)

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


    #Compilar programa estilo SVM usando outros dados, utilizado hinge por dar um menor valor de perda
    model.compile(loss='hinge',
                optimizer='adadelta',
                metrics=['accuracy'])

    model.fit(train_generator, epochs = EPOCHS, validation_data = validation_generator)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Use: transferXception.py <caminho>")

    main(sys.argv[1])