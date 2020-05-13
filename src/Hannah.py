import numpy as np 
import tensorflow as tf 
import os
import matplotlib.image as mpimg
from PIL import Image

from src.training_plots import *
from src.util_images import get_annotated_data
from src.util_images import *

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
    print(tf.shape(u9))
    c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', \
        kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', \
        kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1,1), activation ='sigmoid')(c9)
    print(tf.size(outputs))
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='cosine_similarity', metrics=['accuracy'])
    #on peut tester avec adam et avec stochastic grad. descent
    
    model.summary()
    #Définit X et Y !!
    results = model.fit(X, Y, validation_split=0.1, batch_size=32, epochs=25, shuffle=True)
    model.evaluate(X, Y)


    training_curves(results, EPOCHS=25)


    return model


if __name__ == "__main__":
    img_dim = (572,572,1)
    train_test_split = 0.4
    X, Y = get_annotated_data(40, new_size=(572,572))
    X_train, Y_train = X[:30], Y[:30]
    X_test, Y_test = X[30:], Y[30:]


    model = Unet_method(X_train, Y_train, img_dim)
    model.evaluate(X_test, Y_test)

    X, Y = get_annotated_data(5, new_size=(572,572), show_images=True)
    plot_image(X[0])


    plot_image(image_with_mask(X_train[0], Y_train[0]))
    image = X_test[0]
    mask = Y_test[0]
    pred_mask = model.predict(X_test)[0]
    plot_image(image_with_mask(image, mask))
    plot_image(image_with_mask(image, pred_mask))

    """test display branch"""