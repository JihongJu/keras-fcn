import keras
import keras.backend as K
from keras.losses import categorical_crossentropy


def categorical_crossentropy(y_true, y_pred):
    return K.mean(keras.losses.categorical_crossentropy(y_pred, y_true),
                  axis=[1, 2])
