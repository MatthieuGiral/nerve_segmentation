import numpy as np 
import tensorflow as tf 
import os
import matplotlib.image as mpimg
from PIL import Image
from training_plots import *
from util_images import get_annotated_data
from util_images import *

def Unet_method(X,Y,img_dim):
    [img_width, img_depth, img_channels] = img_dim

    inputs = tf.keras.layers.Input((img_width, img_depth, img_channels))
    #c1 only take float inputs so we multiply pixels by the values in the weights 
    #to make them float (between 0-1)
    s = tf.keras.layers.Lambda(lambda x: x/255)(inputs) #float
    c1 = tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', \
        kernel_initializer='he_normal', padding='same')(s)
    #padding = same: size input image = size output image
    #conv2D ? 16, (3,3) ?

    ## encoding path:
    c1 = tf.keras.layers.Dropout(0.1)(c1) #avoids overfitting
    c1 = tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', \
    kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', \
        kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', \
    kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', \
        kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.1)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', \
    kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', \
        kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', \
    kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

    #dropout increases with layers -> à tester

    c5 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', \
        kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', \
    kernel_initializer='he_normal', padding='same')(c5)

    ##decoding path:
    #   now dropout decreases again
    u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), \
        strides=(2,2),padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', \
        kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', \
        kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), \
        strides=(2,2),padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', \
        kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', \
        kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(16, (2,2), \
        strides=(2,2),padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', \
        kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', \
        kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), \
        strides=(2,2),padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    #print(tf.shape(u9))
    c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', \
        kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', \
        kernel_initializer='he_normal', padding='same')(c9)

    c10 = tf.keras.layers.Conv2D(2, (1,1), activation ='sigmoid')(c9)
    outputs = tf.keras.layers.Softmax(axis = 3)(c10)
    #print(tf.size(outputs))
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    #on peut tester avec adam et avec stochastic grad. descent
    
    model.summary()
    #Définit X et Y !!
    results = model.fit(X, Y, validation_split=0.1, batch_size=10, epochs=10, shuffle=True)
    model.evaluate(X, Y)


    training_curves(results, EPOCHS=10)


    return model

# Définition de notre metrique, exemple avecdice coef :
def dice_coeff (y_true_temp, y_pred, smooth = 1):
    #print("ytrue: ", y_true_temp)
    #print("", "y pred:", y_pred)
    y_true = tf.cast(y_true_temp, dtype = 'float32')
    #print("ytrue: ", y_true)

    numerator = 2.0 * tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2))

    dice = 1-(numerator + smooth)/(denominator + smooth)
    return dice

#définition de notre loss function
def binary_loss (y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy() #binary cross entropy with logits ?
    return bce(y_true, y_pred)

def loss_function (y_true, y_pred):
    return (binary_loss(y_true,y_pred)+dice_coeff(y_true,y_pred))



if __name__ == "__main__":
    img_dim = (544,544,1)
    train_test_split = 0.4
    X, Y = get_annotated_data(40, new_size=(544,544))
    #Y1 = np.where(Y_temp == 0, 1, 0)
    #print("Y1: ", Y1.shape)
    #Y2 = np.where(Y_temp == 1, 1, 0)
    #Y = np.concatenate((Y1,Y2), axis= 3)
    #print("")
    #print("Y:")
    #print (Y)
    #print(Y.shape)
    X_train, Y_train = X[:30], Y[:30]
    X_test, Y_test = X[30:], Y[30:]
    
    model = Unet_method(X_train, Y_train, img_dim)
    model.evaluate(X_test, Y_test)
    #print(model.predict(X_test[0]))



