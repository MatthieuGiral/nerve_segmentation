import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np


def dice_coeff(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coeff_loss(y_true, y_pred, smooth=1) :
    return -dice_coeff(y_true,y_pred,smooth)

def dice_coeff_generator (factor) :
    def dice_coeff(y_true, y_pred, smooth=1):
        y_true_f=K.flatten(y_true)
        y_pred_f=K.flatten(y_pred)
        intersection=K.sum(y_true_f*y_pred_f)
        return (2.*intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)+smooth)

        # y_pred = tf.reshape(y_pred[:, :, :, 1:], (-1, 256, 256, 1))
        # numerator = 2.0 * tf.reduce_sum(y_true * y_pred, axis=(1, 2))
        # denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2))
        # x=1
        # weighting = x * factor + 1
        # dice = weighting*(numerator + smooth)/(denominator + smooth)
        #
        # return dice
        # intersection = K.sum(K.sum(y_true * y_pred, axis=-1), axis=-1)
        # sum_pred = K.sum(K.sum(y_pred, axis=-1), axis=-1)
        # sum_true = K.sum(K.sum(y_true, axis=-1), axis=-1)
        # if K.greater_equal(sum_pred,1):
        #     x= 1
        # else :
        #     x=0
        # weighting = factor*x+1
        # return K.mean(weighting * (2. * intersection + smooth) / (sum_true + sum_pred + smooth))
    return dice_coeff

def dice_loss_generator (factor) :
    def dice_loss(y_true, y_pred, smooth=1) :
        return 1-dice_coeff_generator(factor)(y_true,y_pred,smooth)
    return dice_loss

def sum_dice_cross_entropy (y_true, y_pred):
    return (tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)(y_true,y_pred)+1-dice_coeff(y_true,y_pred))


# def dice_coeff (y_true, y_pred, smooth = 1, factor):
#     intersection = K.sum(K.sum(y_true * y_pred, axis=-1), axis=-1)
#     sum_pred = K.sum(K.sum(y_pred, axis=-1), axis=-1)
#     sum_true = K.sum(K.sum(y_true, axis=-1), axis=-1)
#
#     weighting = K.greater_equal(sum_true, 1) * factor + 1
#     return K.mean(weighting * (2. * intersection + smooth) / (sum_true + sum_pred + smooth))

    # y_pred = tf.reshape(y_pred[:,:,:,1:], (-1,544,544,1))
    # numerator = 2.0 * tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    # denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2))
    # sum_true = tf.reduce_sum(y_true+y_pred)
    # weighting = K.greater_equal(sum_true, 1) * factor + 1
    # dice = (numerator + smooth)/(denominator + smooth)
    #
    # return dice