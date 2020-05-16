import numpy as np
import tensorflow as tf
import warnings
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
                 loss_f = sum_dice_cross_entropy,
                 param_conv = { 'dropout': False,
                                'activation' : 'relu',
                                'kernel_initializer' : 'he_normal',
                                'padding' : 'same'}
                 ):
        self.img_dims = img_dims
        self.param_conv = param_conv
        self.architecture = {'encoding_path': list(reversed(architecture))[:-1],
                             'bottom': architecture[-1],
                             'decoding_path': architecture[1:]}
        self.depth = len(architecture)
        self.loss_function = loss_f
        self.is_trained = False

        self.model = self.construct_network()
        return

    @staticmethod
    def convolution_process(in_tensor, filters, dropout = False, **kwargs):
        print(in_tensor)
        print(kwargs['padding'])
        c = tf.keras.layers.Conv2D(filters, (3, 3),
                                   padding = kwargs['padding'],
                                   activation=kwargs['activation'],
                                   kernel_initializer=kwargs['kernel_initializer'])(in_tensor)
        if dropout is not False:
            c = tf.keras.layers.Dropout(dropout)(c)
        c = tf.keras.layers.Conv2D(filters, (3, 3),
                                   padding = kwargs['padding'],
                                   activation=kwargs['activation'],
                                   kernel_initializer=kwargs['kernel_initializer'])(c)
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
             segmenter.convolution_process(intermediate_tensors_before_conv[-1],
                                           self.architecture['bottom'],
                                           **self.param_conv))

        #Decoding_path
        for i in range(len(self.architecture['decoding_path'])):
            intermediate_tensors_before_conv.append(
                segmenter.concat_process(intermediate_tensors_after_conv[-1],
                                         intermediate_tensors_after_conv[self.depth - 2 - i],
                                         self.architecture['decoding_path'][i]))
            intermediate_tensors_after_conv.append(
                segmenter.convolution_process(intermediate_tensors_before_conv[-1],
                                              self.architecture['decoding_path'][i],
                                              **self.param_conv))

        outputs = tf.keras.layers.Conv2D(2, (1, 1), activation='sigmoid')(intermediate_tensors_after_conv[-1])

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss=self.loss_function, metrics=[dice_coeff])
        model.summary()

        return model

    def train(self, X,Y, epochs, batch_size):
        results = self.model.fit(X, Y, validation_split=0.1, batch_size=batch_size, epochs=epochs) #, callbacks=callbacks
        training_curves(results)
        self.is_trained=True
        return results

    def evaluate(self,X,Y, display_prediction=False):
        """ Evaluate the network on X and Y and display 5 random mask predictions"""
        if self.is_trained==False :
            warnings.warn("Networks Has not been trained")
        # evaluation=self.model.evaluate(X,Y, batch_size=2)
        if display_prediction==True :
            n_data=X.shape[0]
            Random_indices= np.random.randint(low = 0, high= n_data,size =5)
            X2=X[Random_indices]
            Y2=Y[Random_indices]
            predict_example_and_plot(self.model,X2,Y2)
        return evaluation


if __name__ == '__main__':
    print("a jour2")
    # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
    # 
    # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    img_dim = (480, 480, 1)
    test_split = 0.2
    n_images=30
    X_train, Y_train, X_test, Y_test = Training_and_test_batch(n_images,test_split, new_size=(544,544), show_images=False)
    unet = segmenter([512,256,128,64],loss_f= sum_dice_cross_entropy)
    unet.train(X_train,Y_train, epochs=20, batch_size=5)
    # unet.evaluate(X_test,Y_test,display_prediction=True)