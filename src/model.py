import numpy as np
import tensorflow as tf
import os
import matplotlib.image as mpimg
from PIL import Image
try:
    from src.training_plots import *
    from src.util_images import get_annotated_data
    from src.util_images import *
    from src.metrics import *
except:
    from training_plots import *
    from util_images import get_annotated_data
    from util_images import *
    from metrics import *

class segmenter():

    def __init__(self,
                 architecture = [1024,512,256,128,64],
                 img_dims = (544,544,1),
                 dropout = False,
                 loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                 param_conv = { 'dropout': False,
                                'activation' : 'relu',
                                'kernel_initializer' : 'he_normal',
                                'padding' : 'same'}
                 ):
        self.img_dims = img_dims
        self.param_conv = param_conv
        self.architecture = {'encoding_path': architecture.reverse()[:-1],
                             'bottom': architecture[-1],
                             'decoding_path': architecture[:1]}
        self.depth = len(architecture)
        self.model = self.construct_network()
        return

    @staticmethod
    def convolution_process(in_tensor, filters, dropout = False, **kwargs):
        c = tf.keras.layers.Conv2D(filters, (3, 3), **kwargs)(
            in_tensor)
        if dropout is not False:
            c = tf.keras.layers.Dropout(dropout)(c)
        c = tf.keras.layers.Conv2D(filters, (3, 3), kwargs)(c)
        return c

    @staticmethod
    def concat_process(tensor_1, tensor_2, n_filters):
        return tf.keras.layers.concatenate([
                    tf.keras.layers.Conv2DTranspose(n_filters, (2, 2),
                                            strides=(2, 2), padding='same')(tensor_1),
                    tensor_2])

    def construct_network(self):
        [img_width, img_depth, img_channels] = self.img_dims
        inputs = tf.keras.layers.Input((img_width, img_depth, img_channels))
        intermediate_tensors_before_conv = [inputs]
        intermediate_tensors_after_conv = []

        # Encoding path
        for n_filters in self.architecture['encoding_path']:
            intermediate_tensors_after_conv.append(
                segmenter.convolution_process(intermediate_tensors_before_conv[-1],
                                                                           n_filters,
                                                                           **self.param_conv))
            intermediate_tensors_before_conv.append(
                tf.keras.layers.MaxPooling2D((2,2))(intermediate_tensors_after_conv[-1]))

        # Bottom
        intermediate_tensors_after_conv.append(
             segmenter.convolution_process(intermediate_tensors_before_conv,
                                           self.architecture['bottom'],
                                           **self.param_conv))

        #Decoding_path
        for i in range(len(self.architecture['decoding_path'])):
            intermediate_tensors_before_conv.append(
                segmenter.concat_process(intermediate_tensors_after_conv[-1],
                                         intermediate_tensors_after_conv[self.depth - 1 - i],
                                         self.architecture['decoding_path'][i]))
            intermediate_tensors_after_conv.append(
                segmenter.convolution_process(intermediate_tensors_before_conv[-1],
                                              self.architecture['decoding_path'][i]))

        outputs = tf.keras.layers.Conv2D(2, (1, 1), activation='sigmoid')(intermediate_tensors_after_conv[-1])

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss=self.loss_function, metrics=[dice_coeff])
        model.summary()

        return model

if __name__ == '__main__':
    img_dim = (544, 544, 1)
    train_test_split = 0.4
    X, Y = get_annotated_data(100, new_size=(544, 544))
    X_train, Y_train = X[:80], Y[:80]
    X_test, Y_test = X[:80], Y[:80]