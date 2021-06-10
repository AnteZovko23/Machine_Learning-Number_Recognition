from tensorflow import  keras

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import KFold
from tensorflow.python.keras.layers.normalization import BatchNormalization

### Overhead ######

## Load data
def load_data():
    numbers_mnist = keras.datasets.mnist

    (train_images, train_labels), (test_images, test_labels) = numbers_mnist.load_data()

    ## Encode labels
    train_labels = keras.utils.to_categorical(train_labels)
    test_labels = keras.utils.to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels

## Preprocess data
def process_data(train_data, test_data):

    train_data = train_data / 255.0
    test_data = test_data / 255.0

    return np.expand_dims(train_data, -1), np.expand_dims(test_data, -1)

## Create Model
def create_model(hidden_number=128, second_hidden_number=64):
    model = keras.Sequential([

        tf.keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=(28, 28, 1)),  
        tf.keras.layers.MaxPooling2D((2, 2)),  
        keras.layers.BatchNormalization(),

        keras.layers.Flatten(),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(hidden_number, activation='relu'),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(second_hidden_number,activation='relu'),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(10, activation='softmax'),

        # keras.layers.Dropout(0.1)

    ])

    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    return model

# Evaluated using k-fold cross-validation
def evaluate_model(data, labels, folds = 5, batch_size=32, epochs=10, hidden_number=128, second_hidden_number=64):
    scores, histories = [], []

    k_fold = KFold(folds, shuffle=True, random_state=1)
    
    model = create_model(hidden_number, second_hidden_number)

    for train_i, test_i in k_fold.split(data):


        ## Select split data
        trainX, trainY, testX, testY = data[train_i], labels[train_i], data[test_i], labels[test_i]

        # Fit model
        history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), verbose=1)

        _, accuracy = model.evaluate(testX, testY, verbose=1)

        scores.append(accuracy)
        histories.append(history)

    save_model(model)
    return scores, histories


## Plot the performance
def plot_performance(scores, histories):
    for i in range(len(histories)):

        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.show()

## Start Training and Evaluation
def start():
    
    # Load Data
    trainImages, trainLabels, testImages, testLabels = load_data()

    # Preprocess
    trainImages, testImages = process_data(trainImages, testImages)

    ## Train Model
    scores, histories = evaluate_model(trainImages, trainLabels, 5, 32, 10, 128, 64)

    plot_performance(scores, histories)

def save_model(model):
    model.save('my_model')

def load_model():
    return keras.models.load_model('my_model')


start()

