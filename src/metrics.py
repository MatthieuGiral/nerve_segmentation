import tensorflow as tf

def dice_coeff (y_true, y_pred, smooth = 1):
    y_pred = tf.reshape(y_pred[:,:,:,1:], (-1,544,544,1))
    numerator = 2.0 * tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2))

    dice = 1-(numerator + smooth)/(denominator + smooth)
    return dice

def sum_dice_cross_entropy (y_true, y_pred):
    return (tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)(y_true,y_pred)-dice_coeff(y_true,y_pred))