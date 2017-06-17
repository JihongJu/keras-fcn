import numpy as np
import keras.backend as K
from keras_fcn.bases import (
    BaseNet,
    VGG16)


def test_basenet():
    assert BaseNet()(3) == [3]


def test_vgg16():
    a = K.variable(np.random.random((1, 28, 28, 28)))
    b, c, d = VGG16()(a)
    if K.image_data_format == 'channels_last':
        assert b.shape == (1, 29, 29, 256)
        assert c.shape == (1, 15, 15, 512)
        assert d.shape == (1, 2, 2, 4096)
    if K.image_data_format == 'channels_last':
        assert b.shape == (1, 256, 29, 29)
        assert c.shape == (1, 512, 15, 15)
        assert d.shape == (1, 4096, 2, 2)
