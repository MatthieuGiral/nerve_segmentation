import matplotlib.pyplot as plt
import tensorflow as tf
try:
    from util_images import plot_image_with_mask
except:
    from src.util_images import plot_image_with_mask
import numpy as np

def training_curves (results) :
    """ Displays accuracy on training and validation batches after each epoch"""

    plt.figure()
    for key in results.history.keys() :
        plt.plot(results.history[key])
        plt.plot(results.history[key], label=key)
        plt.title('Training curves')
        plt.legend()

    plt.show()
    return



def predict_example_and_plot(model, X, Y):
    for i in range(len(X)):
        Y_pred = np.argmax(model.predict(X[i].reshape((1,544, 544, 1))),axis = 3)
        plot_image_with_mask(X[i], Y[i], Y_pred)
    return
