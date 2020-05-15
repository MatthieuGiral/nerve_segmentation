import numpy as np
import tensorflow as tf
import os
import matplotlib.image as mpimg
from PIL import Image

try:
    from src.training_plots import *
    from src.util_images import get_annotated_data
    from src.util_images import *
except:
    from training_plots import *
    from util_images import get_annotated_data
    from util_images import *


def Unet_method(X, Y, img_dim):
    [img_width, img_depth, img_channels] = img_dim

    inputs = tf.keras.layers.Input((img_width, img_depth, img_channels))

    c = tf.keras.layers.Conv2D(2, (1, 1), activation='sigmoid')(inputs)
    outputs = tf.keras.layers.Softmax(axis=3)(c)
    # print(tf.size(outputs))
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer='adam', loss=loss_function,
                  metrics=['accuracy', dice_coeff])
    # on peut tester avec adam et avec stochastic grad. descent

    model.summary()
    # Définit X et Y !!
    results = model.fit(X, Y, validation_split=0.1, batch_size=10, epochs=1, shuffle=True)
    model.evaluate(X, Y)
    print(results.history.keys())
    training_curves(results)

    return model


# Définition de notre metrique, exemple avecdice coef :
def dice_coeff(y_true, y_pred, smooth=1):
    y_pred = tf.reshape(y_pred[:, :, :, 1:], (-1, 544, 544, 1))
    numerator = 2.0 * tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2))

    dice = 1 - (numerator + smooth) / (denominator + smooth)
    return dice


# définition de notre loss function
def binary_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()  # binary cross entropy with logits ?
    return bce(y_true, y_pred)


def loss_function(y_true, y_pred):
    return (binary_loss(y_true, y_pred) + dice_coeff(y_true, y_pred))


if __name__ == "__main__":
    img_dim = (544, 544, 1)
    train_test_split = 0.4
    X, Y = get_annotated_data(40, new_size=(544, 544))
    # Y1 = np.where(Y_temp == 0, 1, 0)
    # print("Y1: ", Y1.shape)
    # Y2 = np.where(Y_temp == 1, 1, 0)
    # Y = np.concatenate((Y1,Y2), axis= 3)
    # print("")
    # print("Y:")
    # print (Y)
    # print(Y.shape)
    X_train, Y_train = X[:5], Y[:5]
    X_test, Y_test = X[30:], Y[30:]

    model = Unet_method(X_train, Y_train, img_dim)
    print(model.history)
    model.evaluate(X_test, Y_test)
    predict_example_and_plot(model, X_train[:3], Y_train[:3])


    # print(model.predict(X_test[0]))
