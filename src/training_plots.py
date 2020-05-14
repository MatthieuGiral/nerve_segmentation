import matplotlib.pyplot as plt
import tensorflow as tf
try:
    from util_images import plot_image_with_mask
except:
    from src.util_images import plot_image_with_mask
import numpy as np

def training_curves (results, EPOCHS) :
    """ Displays accuracy on training and validation batches after each epoch"""
    epochs = range(EPOCHS)

    accuracy = results.history['accuracy']
    val_accuracy = results.history['val_accuracy']

    plt.figure()
    plt.plot(epochs, accuracy, 'r', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'bo', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

def predict_example_and_plot(model, X, Y):
    for i in range(len(X)):
        Y_pred = np.argmax(model.predict(X[i].reshape((1,544, 544, 1))),axis = 3)
        plot_image_with_mask(X[i], Y[i], Y_pred)
    return
