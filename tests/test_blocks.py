import keras.backend as K
from keras.layers import Input
from keras_fcn.blocks import vgg_conv


def test_vgg():
    if K.image_data_format() == 'channels_first':
        x = Input(shape=(3, 224, 224))
        y1_shape = (None, 64, 211, 211)
        y2_shape = (None, 128, 106, 106)
    else:
        x = Input(shape=(224, 224, 3))
        y1_shape = (None, 211, 211, 64)
        y2_shape = (None, 106, 106, 128)

    block1 = vgg_conv(filters=64, convs=2, block_name='block1')
    y = block1(x)
    assert K.int_shape(y) == y1_shape

    block2 = vgg_conv(filters=128, convs=2, block_name='block2')
    y = block2(y)
    assert K.int_shape(y) == y2_shape
