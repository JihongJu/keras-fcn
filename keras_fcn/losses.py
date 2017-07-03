import keras
import keras.backend as K


def mean_categorical_crossentropy(y_true, y_pred):
    loss = K.mean(keras.losses.categorical_crossentropy(y_true, y_pred), axis=[1, 2])
    return loss
