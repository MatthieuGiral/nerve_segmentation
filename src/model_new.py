import numpy as np
import tensorflow as tf
import warnings
import os
import pickle
import matplotlib.image as mpimg
import uuid
from PIL import Image
try:
    from src.training_plots import *
    from src.util_images import get_annotated_data
    from src.util_images import *
except:
    from training_plots import *
    from util_images import get_annotated_data
    from util_images import *

pix = 96
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../models/')

def predict_example_and_plot(model, X, Y, size):
    for i in range(len(X)):
        Y_pred = model.predict(X[i].reshape((1,pix, pix,1))) > 0.5
        plot_image_with_mask(X[i], Y[i], pred_mask=Y_pred, size = pix)
    return


def dice_coef_loss(y_true, y_pred, smooth=1.0):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return -(2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_coef_eval(y_true, y_pred, smooth = 0.01):
    y_pred = tf.cast(y_pred > 0.5, dtype=tf.float32)
    target = tf.cast(y_true > 0.5, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(y_pred, y_true), axis=(1,2,3))
    l = tf.reduce_sum(y_pred, axis=(1,2,3))
    r = tf.reduce_sum(y_true, axis=(1,2,3))
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    hard_dice = tf.reduce_mean(hard_dice, name='hard_dice')
    return hard_dice

class segmenter():

    def __init__(self,
                 architecture = [1024,512,256,128,64],
                 img_dims = (544,544,1),
                 lambd = 1e-5,
                 dropout = False,
                 loss_f = dice_coef_loss,
                 metrics = [],
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
        self.optimizer = tf.keras.optimizers.Adam(lambd)
        self.lambd = lambd
        self.depth = len(architecture)
        self.loss_function = loss_f
        self.is_trained = False
        self.metrics = metrics
        self.id_model = np.random.randint(10000,99999)
        self.model = self.construct_network()
        return

    def save(self):
        file = open(os.path.join(model_dir,f'model_{self.id_model}.pck'), 'wb')

        pickle.dump(self, file)
        file.close()
        file = open(os.path.join(model_dir,'summary.txt'), 'a')
        file.write(f'model:{self.id_model} {self.architecture} {self.img_dims} {self.score}\n')
        file.close()
        return

    def __getstate__(self):
        state = self.__dict__
        del state['model']
        del state['metrics']
        del state['loss_function']
        del state['optimizer']
        state['training_results'] = state['training_results'].history
        return state

    @staticmethod
    def convolution_process(in_tensor, filters, dropout = False, **kwargs):
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

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(intermediate_tensors_after_conv[-1])

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)
        model.summary()

        return model


    def train(self, X, Y, epochs, batch_size):
        results = self.model.fit(X, Y, validation_split=0.1,
                                 batch_size=batch_size,
                                 epochs=epochs) #, callbacks=callbacks
        self.training_results = results
        training_curves(results)
        self.is_trained=True
        return results

    def evaluate(self,X,Y, display_prediction=False):
        """ Evaluate the network on X and Y and display 5 random mask predictions"""
        if self.is_trained==False :
            warnings.warn("Networks Has not been trained")
        self.score = self.model.evaluate(X,Y)
        print(self.score)
        if display_prediction==True :
            n_data=X.shape[0]
            Random_indices= np.random.randint(low = 0, high= n_data,size =5)
            X2=X[Random_indices]
            Y2=Y[Random_indices]
            predict_example_and_plot(self.model,X2,Y2, size = self.img_dims[0])
        return


if __name__ == '__main__':
    img_dim = (96, 96, 1)
    test_split = 0.2
    n_images = 5000
    X_train, Y_train, X_test, Y_test = Training_and_test_batch(n_images,test_split, new_size=(96,96), show_images=False)
    for size in [64,96,128,160]:
        try:
            X_train, Y_train, X_test, Y_test = Training_and_test_batch(n_images, test_split, new_size=(size, size),
                                                                       show_images=False)
            unet = segmenter([1024,512,256,128,64],
                             (size,size,1),
                             loss_f=dice_coef_loss,
                             lambd=1e-5,
                             metrics = [dice_coef_eval])
            unet.train(X_train,Y_train, epochs=20, batch_size=64)
            unet.evaluate(X_test,Y_test,display_prediction=False)
            unet.save()
        except:
            print('error')
    X_train, Y_train, X_test, Y_test = Training_and_test_batch(n_images, test_split, new_size=(96, 96),
                                                               show_images=False)
    for lam in [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]:
        try:
            unet = segmenter([1024,512,256,128,64],
                             img_dim,
                             loss_f=dice_coef_loss,
                             lambd=lam,
                             metrics = [dice_coef_eval])
            unet.train(X_train,Y_train, epochs=20, batch_size=64)
            unet.evaluate(X_test,Y_test,display_prediction=False)
            unet.save()
        except:
            print('error')


