import matplotlib.pyplot as plt
import tensorflow as tf

def training_curves (results, EPOCHS) :
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
