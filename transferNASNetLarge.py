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
                                       validation_split=0.2,
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
    outputs = keras.layers.Dense(12, kernel_regularizer=keras.regularizers.L2(0.01), activation='linear')(x)

    model = keras.Model(inputs, outputs)

    model.summary()

    model.compile(
        loss='hinge',
        optimizer='adadelta',
        metrics=['accuracy']
    )

    history = model.fit(train_generator, epochs = EPOCHS, validation_data = validation_generator)

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
    plt.title('Training and validation accuracy - Transfer EfficientNetB2')
    plt.legend()
    plt.savefig('Resultados/Imagens/Transfer1_Acc_' + caminho)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss - Transfer EfficientNetB2' + caminho)
    plt.legend()
    plt.savefig('Resultados/Imagens/Transfer1_Loss_' + caminho)

    base_model.trainable = True
    model.summary()

    #Compilar programa estilo SVM usando outros dados, utilizado hinge por dar um menor valor de perda
    model.compile(loss='hinge',
                optimizer='adadelta',
                metrics=['accuracy'])

    history = model.fit(train_generator, epochs = EPOCHS, validation_data = validation_generator)

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
    plt.title('Training and validation accuracy - Transfer EfficientNetB2 Fine Tuning')
    plt.legend()
    plt.savefig('Resultados/Imagens/Transfer1FT_Acc_' + caminho)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss - Transfer EfficientNetB2 Fine Tuning' + caminho)
    plt.legend()
    plt.savefig('Resultados/Imagens/Transfer1FT_Loss_' + caminho)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Use: transferXception.py <caminho>")

    main(sys.argv[1])