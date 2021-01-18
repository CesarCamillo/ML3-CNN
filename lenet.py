#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import numpy as np
import sys
import os

from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import keras
import keras.layers as layers
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard

# %matplotlib inline
import matplotlib.pyplot as plt

def main(caminho):

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
        class_mode='categorical',
        subset='training',
        color_mode='grayscale'
    )

    validation_generator = train_datagen.flow_from_directory(
        directory = caminho, # same directory as training data
        target_size=(32, 32),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        color_mode='grayscale'
    )

    model = keras.Sequential()

    model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
    model.add(layers.AveragePooling2D())

    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(layers.AveragePooling2D())

    model.add(layers.Flatten())

    model.add(layers.Dense(units=120, activation='relu'))

    model.add(layers.Dense(units=84, activation='relu'))

    model.add(layers.Dense(units=12, activation = 'softmax'))

    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])    

    history = model.fit(
                train_generator,
                steps_per_epoch = train_generator.samples // BATCH_SIZE,
                validation_data = validation_generator, 
                validation_steps = validation_generator.samples // BATCH_SIZE,
                epochs = EPOCHS)

    Y_pred = model.predict_generator(validation_generator, validation_generator.samples  // BATCH_SIZE+1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, y_pred))
    print('Classification Report')
    print(classification_report(validation_generator.classes, y_pred))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy - LeNet 5')
    plt.legend()
    plt.savefig('Resultados/Imagens/LeNet5_Acc_' + caminho)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss - LeNet 5' + caminho)
    plt.legend()
    plt.savefig('Resultados/LeNet5_Loss_' + caminho)

if __name__ == "__main__":
    if len(sys.argv) != 2:
            sys.exit("Use: lenet.py <caminho>")

    main(sys.argv[1])


