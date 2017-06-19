import pytest
import numpy as np
import keras.backend as K
from keras.layers import Input
from keras_fcn.decoders import (
    Decoder,
    VGGDecoder
)

from keras.utils.test_utils import keras_test


def test_decoder():
    with pytest.raises(ValueError):
        Decoder(pyramid=['fake1', 'fake2'], blocks=['fake1'])


def test_vgg_decoder():
    if K.image_data_format() == 'channels_last':
        inputs = Input(shape=(500, 500, 3))
        pool3 = Input(shape=(88, 88, 256))
        pool4 = Input(shape=(44, 44, 512))
        drop7 = Input(shape=(16, 16, 4096))
        score_shape = (None, 500, 500, 21)
    else:
        inputs = Input(shape=(3, 500, 500))
        pool3 = Input(shape=(256, 88, 88))
        pool4 = Input(shape=(512, 44, 44))
        drop7 = Input(shape=(4096, 16, 16))
        score_shape = (None, 21, 500, 500)
    pyramid = [drop7, pool4, pool3, inputs]
    scales = [1., 1e-2, 1e-4]
    score = VGGDecoder(pyramid, scales, classes=21)
    assert K.int_shape(score) == score_shape
