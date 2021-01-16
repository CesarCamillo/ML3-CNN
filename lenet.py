#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import numpy as np
import sys
import os

from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import keras
import keras.layers as layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

def main(caminho):
    sns.set()    

    EPOCHS = 20
    BATCH_SIZE = 128

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        directory = caminho,
        target_size=(32,32),
        batch_size=BATCH_SIZE,
        class_mode=None,
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        directory = caminho, # same directory as training data
        target_size=(32, 32),
        batch_size=BATCH_SIZE,
        class_mode=None,
        subset='validation')

    # model = keras.Sequential()

    # model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
    # model.add(layers.AveragePooling2D())

    # model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    # model.add(layers.AveragePooling2D())

    # model.add(layers.Flatten())

    # model.add(layers.Dense(units=120, activation='relu'))

    # model.add(layers.Dense(units=84, activation='relu'))

    # model.add(layers.Dense(units=10, activation = 'softmax'))

    # model.summary()

    # model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])    

    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch = train_generator.samples // BATCH_SIZE,
    #     validation_data = validation_generator, 
    #     validation_steps = validation_generator.samples // BATCH_SIZE,
    #     epochs = EPOCHS)

    # Y_pred = model.predict_generator(validation_generator, num_of_test_samples // batch_size+1)
    # y_pred = np.argmax(Y_pred, axis=1)
    # print('Confusion Matrix')
    # print(confusion_matrix(validation_generator.classes, y_pred))
    # print('Classification Report')
    # print(classification_report(validation_generator.classes, y_pred))

if __name__ == "__main__":
    if len(sys.argv) != 2:
            sys.exit("Use: lenet.py <caminho>")

    main(sys.argv[1])


