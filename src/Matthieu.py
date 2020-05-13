import Hannah as H

from training_plots import *
from util_images import get_annotated_data
from util_images import *

class U_net():

    def __init__(self,
                 img_dims):
        self.img_dims = img_dims
        self.construct_network()
        return

    def construct_network(self):
        [img_width, img_depth, img_channels] = self.img_dims

        inputs = tf.keras.layers.Input((img_width, img_depth, img_channels))
        # c1 only take float inputs so we multiply pixels by the values in the weights
        # to make them float (between 0-1)
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)  # float

        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        # c1=c1[:,:570,:570,:]
        print(c1.shape)
        c1 = tf.keras.layers.Dropout(0.1)(c1)

        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(c1)
        # c1 = c1[:, :568, :568, :]
        p1 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(c1)
        print(p1.shape)
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(p1)
        # c1 = c1[:, :282, :282, :]
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(c2)
        # c2 = c2[:, :280, :280, :]
        print(c2.shape)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(p2)
        # c3 = c3[:, :138, :138, :]
        c3 = tf.keras.layers.Dropout(0.1)(c3)
        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(c3)
        # c3 = c3[:, :136, :136, :]
        print(c3.shape)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)



        # dropout increases with layers -> à tester


        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(p3)
        # c4 = c4[:, :66, :66, :]
        c4 = tf.keras.layers.Dropout(0.15)(c4)
        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(c4)
        # c4 = c4[:, :64, :64,:]
        print(c4.shape)
        p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

        c5 =  tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(p4)
        # c5 = c5[:, :30, :30, :]
        c5 = tf.keras.layers.Dropout(0.2)(c5)

        c5 =  tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', \
                                    kernel_initializer='he_normal', padding='same')(c5)
        # c5 = c5[:, :28, :28, :]

        ##decoding path:
        #   now dropout decreases again

        u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), \
                                             strides=(2, 2), padding='same')(c5)
        print(u6.shape)
        u6 = tf.keras.layers.concatenate([u6, c4])
        print(u6.shape)
        c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', \
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
        model.compile(optimizer='adam', loss='cosine_similarity', metrics=['accuracy'])
        model.summary()

if __name__ == '__main__':
    U_net((544,544,1))