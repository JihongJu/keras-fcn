import numpy as np
import keras.backend as K
from keras_fcn.losses import categorical_crossentropy


def test_categorical_crossentropy():

    y_true = np.reshape([1, 1, 0, 0], [1, 2, 2]).astype('int')
    y_true = np.eye(2)[y_true]
    y_pred = np.ones((1, 2, 2, 2)) * 0.5

    y_true, y_pred = K.variable(y_true), K.variable(y_pred)

    loss = categorical_crossentropy(y_true, y_pred)
    loss = K.get_value(loss)
    assert np.allclose(loss, 8.0590477)
