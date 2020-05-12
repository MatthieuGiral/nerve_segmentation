import numpy as np 
import tensorflow as tf 
import os
import matplotlib.image as mpimg
from PIL import Image

if __name__ == "__main__":

    print("test")
    #Paramètres: 
    n_train = 10 # nb images to load from train file
    """
    # Import images

    cd = os.getcwd()
    file_test = os.path.join(cd, "..\data\test")
    train_test = os.path.join(cd, "..\data\train")

    #path_test = os.path.join(file_test, "*")
    #path_train = os.path.join(file_train,"*")

    data_train = []
    for i in range (n_train):
        data_train.append(Image.open(path_test)) #liste d'images
    # définir l'input -> mettre sous forme d'une matrice
    # quelles dimensions ?
    """
    img_dim = [1, 2, 3]

    Unet_method(X, Y, img_dim)


def Unet_method(X,Y,img_dim):
    [img_width, img_depth, img_channels] = img_dim

    inputs = tf.keras.layers.Input((img_width, img_depth, img_channels))
    #c1 only take float inputs so we multiply pixels by the values in the weights 
    #to make them float (between 0-1)
    s = tf.keras.layers.Lambda(lambda x: x/255)(inputs) #float
    c1 = tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', \
        kernel_initializer='he_normal', padding='same')(s)
    # padding = same: size input image = size output image
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
    c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', \
        kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', \
        kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1,1), activation ='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outpus=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #on peut tester avec adam et avec stochastic grad. descent
    
    model.summary()

    #Définit X et Y !!
    results = model.fit(X, Y, validation_split=0.1, batch_size=16, epochs=25,callbacks=callbacks)