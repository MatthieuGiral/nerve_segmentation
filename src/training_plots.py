import matplotlib.pyplot as plt
import tensorflow as tf
from src.util_images import plot_image_with_mask

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
        plot_image_with_mask(X[i], Y[i], model.predict(X[i]))
