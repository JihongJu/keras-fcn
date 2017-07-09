import keras
import keras.backend as K


def mean_categorical_crossentropy(y_true, y_pred):
    if K.image_data_format() == 'channels_last':
        loss = K.mean(keras.losses.categorical_crossentropy(y_true, y_pred), axis=[1, 2])
    elif K.image_data_format() == 'channels_first':
        loss = K.mean(keras.losses.categorical_crossentropy(y_true, y_pred), axis=[2, 3])
    return loss

def flatten_categorical_crossentropy(classes):
    def f(y_true, y_pred):
        y_true = K.reshape(y_true, (-1, classes))
        y_pred = K.reshape(y_pred, (-1, classes))
        return keras.losses.categorical_crossentropy(y_true, y_pred)
    return f

