from keras.datasets import fashion_mnist
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

def main(caminho):
    EPOCHS = 20
    BATCH_SIZE = 128
    NUM_CLASSES = 12
    LINE_NUM = 128
    COL_NUM = 64

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        directory = caminho,
        target_size=(LINE_NUM,COL_NUM),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        color_mode='grayscale'
    )

    validation_generator = train_datagen.flow_from_directory(
        directory = caminho, # same directory as training data
        target_size=(LINE_NUM, COL_NUM),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        color_mode='grayscale'
    )

    # Find the unique numbers from the train labels
    

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(LINE_NUM, COL_NUM,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))           
    model.add(Dropout(0.3))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    train_dropout = model.fit(
        train_generator,
        steps_per_epoch = train_generator.samples // BATCH_SIZE,
        validation_data = validation_generator, 
        validation_steps = validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS)

    model.summary()

    predicted_classes = model.predict(train_generator)
    predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
    print(predicted_classes.shape)

    Y_pred = model.predict_generator(validation_generator, validation_generator.samples  // BATCH_SIZE+1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, y_pred))
    print('Classification Report')
    print(classification_report(validation_generator.classes, y_pred))

    test_eval = model.evaluate(train_generator)

    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])

    accuracy = train_dropout.history['accuracy']
    val_accuracy = train_dropout.history['val_accuracy']
    loss = train_dropout.history['loss']
    val_loss = train_dropout.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    model.save("model_dropout.h5py")

    # correct = np.where(predicted_classes==test_Y)[0]
    # print("Found %d correct labels" % len(correct))
    # for i, correct in enumerate(correct[:9]):
    #     plt.subplot(3,3,i+1)
    #     plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
    #     plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    #     plt.tight_layout()

    # incorrect = np.where(predicted_classes!=test_Y)[0]
    # print("Found %d incorrect labels" % len(incorrect))
    # for i, incorrect in enumerate(incorrect[:9]):
    #     plt.subplot(3,3,i+1)
    #     plt.imshow(test_X[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    #     plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
    #     plt.tight_layout()


    # plt.show()

    # target_names = ["Class {}".format(i) for i in range(num_classes)]
    # print(classification_report(test_Y, predicted_classes, target_names=target_names))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Use: lenet.py <caminho>")

    main(sys.argv[1])