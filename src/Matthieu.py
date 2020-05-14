import Hannah as H
import tensorflow as tf
from training_plots import *
from util_images import get_annotated_data
from util_images import *

class U_net():

    def __init__(self,
                 img_dims):
        self.img_dims = img_dims
        self.model = self.construct_network()
        return

    def construct_network(self):
        [img_width, img_depth, img_channels] = self.img_dims

        inputs = tf.keras.layers.Input((img_width, img_depth, img_channels))
        # c1 only take float inputs so we multiply pixels by the values in the weights
        # to make them float (between 0-1)
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)  # float

        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        print(c1.shape)
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(c1)
        print(c1.shape)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        print(p1.shape)
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Dropout(0.1)(c3)
        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)



        # dropout increases with layers -> à tester


        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.Dropout(0.15)(c4)
        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(c4)

        p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

        c5 =  tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(p4)

        c5 = tf.keras.layers.Dropout(0.2)(c5)

        c5 =  tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(c5)

        ##decoding path:
        #   now dropout decreases again

        u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), \
                                             strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])

        c6=tf.keras.layers.Conv2D(512, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Dropout(0.15)(c6)
        c6=tf.keras.layers.Conv2D(512, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(c6)

        u7= tf.keras.layers.Conv2DTranspose(256, (2, 2), \
                                             strides=(2, 2), padding='same')(c6)
        u7=tf.keras.layers.concatenate([u7,c3])

        c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(u7)
        c7 = tf.keras.layers.Dropout(0.10)(c7)
        c7= tf.keras.layers.Conv2D(256, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(c7)


        u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), \
                                             strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(u8)
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(c8)

        u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), \
                                             strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1])


        c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(c9)


        outputs = tf.keras.layers.Conv2D(2, (1, 1), activation='sigmoid')(c9)
        print(tf.size(outputs))
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss=loss_function, metrics=[dice_coef])
        model.summary()
        return model

# Définition de notre metrique, exemple avecdice coef :
def dice_coef (y_true, y_pred, smooth = 1):
    #if not len(y_pred) or not len(y_true): return 0.0
    if not y_pred.shape[0] or not y_true.shape[0]: return 0.0
    if y_pred == y_true: return 1.0
    numerator = 2.0 * tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    denominator = tf.reduce_sum(y_true, axis=(1,2)) + \
        tf.reduce_sum(y_pred, axis=(1,2))
    #denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
    #return 1 - numerator / denominator
    
    dice = (numerator + smooth)/(denominator + smooth)
    return dice

#définition de notre loss function
def loss_function (y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    return bce(y_true, y_pred)


if __name__ == '__main__':
    img_dim = (544, 544, 1)
    unet = U_net(img_dim)
    # train_test_split = 0.4
    # n_sample = 50
    # n_train = int(train_test_split*n_sample)
    # X, Y = get_annotated_data(n_sample, new_size=img_dim[:-1])
    # X_train, Y_train = X[:n_train], Y[:n_train]
    # X_test, Y_test = X[n_train:], Y[n_train:]
    # unet.model.fit( X_train, Y_train, validation_split=0.1, batch_size=4, epochs=10)
    # unet.model.evaluate(X_test,Y_test)