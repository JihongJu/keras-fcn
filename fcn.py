from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)
import six
from keras.models import Model
from keras.layers import (
    Input,
    Dropout,
    Lambda,
    merge
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    Deconvolution2D,
    ZeroPadding2D
)
from keras import backend as K


def _deconv(**deconv_params):
    def f(input):
        """Helper to build Deconvolution2D layer
        """
        nb_filter = deconv_params["nb_filter"]
        nb_row = deconv_params["nb_row"]
        nb_col = deconv_params["nb_col"]
        subsample = deconv_params.setdefault("subsample", (2, 2))
        border_mode = deconv_params.setdefault("border_mode", "same")
        name = deconv_params.setdefault("name", "deconv")
        width = input._keras_shape[ROW_AXIS]
        height = input._keras_shape[COL_AXIS]
        if K.image_dim_ordering() == 'tf':
            output_shape = (None, width, height, nb_filter)
        else:
            output_shape = (None, nb_filter, width, height)
        deconv = Deconvolution2D(nb_filter=nb_filter, nb_row=nb_row,
                                 nb_col=nb_col, output_shape=output_shape,
                                 border_mode=border_mode, bias=False,
                                 subsample=subsample, activation='linear',
                                 name=name)(input)
        return deconv

    return f


def vgg8(skip=True):
    """VGG8 as basenet
        # Arguments
        skip: boolean if including the skip architecture of FCN.
    # Returns
        pool1: (only when skip==True) layer with output_shape = input_shape / 2
        pool2: (only when skip==True)layer with output_shape = input_shape / 4
        drop2: layer with output_shape = input_shape / 8
    """
    def f(input):
        return

    return f


def vgg16(skip=True):
    """VGG16 as base net
    # Arguments
        skip: boolean if including the skip architecture of FCN.
    # Returns
        pool3: (only when skip==True) layer with output_shape = input_shape / 8
        pool4: (only when skip==True)layer with output_shape = input_shape / 16
        drop7: layer with output_shape = input_shape / 32
    """
    def f(input):
        conv1_1 = Convolution2D(64, 3, 3, activation='relu',
                                border_mode='same',
                                name='conv1_1')(input)
        conv1_2 = Convolution2D(64, 3, 3, activation='relu',
                                border_mode='same', name='conv1_2')(conv1_1)
        pool1 = MaxPooling2D((2, 2), strides=(2, 2),
                             name='block1_pool')(conv1_2)
        # Block 2
        conv2_1 = Convolution2D(128, 3, 3, activation='relu',
                                border_mode='same',
                                name='conv2_1')(pool1)
        conv2_2 = Convolution2D(128, 3, 3, activation='relu',
                                border_mode='same', name='conv2_2')(conv2_1)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2),
                             name='block2_pool')(conv2_2)
        # Block 3
        conv3_1 = Convolution2D(256, 3, 3, activation='relu',
                                border_mode='same', name='conv3_1')(pool2)
        conv3_2 = Convolution2D(256, 3, 3, activation='relu',
                                border_mode='same', name='conv3_2')(conv3_1)
        conv3_3 = Convolution2D(256, 3, 3, activation='relu',
                                border_mode='same', name='conv3_3')(conv3_2)
        pool3 = MaxPooling2D((2, 2), strides=(2, 2),
                             name='block3_pool')(conv3_3)
        # Block 4
        conv4_1 = Convolution2D(512, 3, 3, activation='relu',
                                border_mode='same', name='conv4_1')(pool3)
        conv4_2 = Convolution2D(512, 3, 3, activation='relu',
                                border_mode='same', name='conv4_2')(conv4_1)
        conv4_3 = Convolution2D(512, 3, 3, activation='relu',
                                border_mode='same', name='conv4_3')(conv4_2)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2),
                             name='block4_pool')(conv4_3)
        # Block 5
        conv5_1 = Convolution2D(512, 3, 3, activation='relu',
                                border_mode='same', name='conv5_1')(pool4)
        conv5_2 = Convolution2D(512, 3, 3, activation='relu',
                                border_mode='same', name='conv5_2')(conv5_1)
        conv5_3 = Convolution2D(512, 3, 3, activation='relu',
                                border_mode='same', name='conv5_3')(conv5_2)
        pool5 = MaxPooling2D((2, 2), strides=(2, 2),
                             name='block5_pool')(conv5_3)
        # fully conv
        fc6 = Convolution2D(4096, 7, 7, activation='relu',
                            border_mode='same', name='fc6')(pool5)
        drop6 = Dropout(0.5)(fc6)
        fc7 = Convolution2D(4096, 1, 1, activation='relu',
                            border_mode='valid', name='fc7')(drop6)
        drop7 = Dropout(0.5)(fc7)

        if skip:
            return (pool3, pool4, drop7)
        else:
            return drop7

    return f


def _get_basenet(identifier):
    if isinstance(identifier, six.string_types):
        basenet = globals().get(identifier.lower())
        if not basenet:
            raise ValueError('Invalid {}'.format(identifier))
        return basenet
    return identifier


def _handle_dim_ordering():
    """dim_ordering handler by @raghakot
    (See https://github.com/raghakot/keras-resnet/blob/master/resnet.py)
    """
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def FCN(basenet='vgg16', weights=None, num_output=21,
        input_shape=(None, None, 3)):
    """Instantiate the FCN8s architecture with keras
    # Arguments
        weights: pre-trained models
        num_output: number of classes
        input_shape: input image shape (Support only 32, 64, 128, ... for
        now)
    # Returns
        A Keras model instance
    """
    _handle_dim_ordering()
    basenet = _get_basenet(basenet)
    # input
    input = Input(shape=input_shape)
    # the base net: VGG16
    pool3, pool4, drop7 = basenet(skip=True)(input)

    score_fr = Convolution2D(num_output, 1, 1, activation='linear',
                             border_mode='valid', name='score_fr')(drop7)
    _, score_fr_dim_row, score_fr_dim_col, _ = score_fr.shape
    upscore2_output_shape = (None,
                             2 * int(score_fr_dim_row),
                             2 * int(score_fr_dim_col),
                             num_output)
    upscore2 = Deconvolution2D(num_output, 4, 4,
                               output_shape=upscore2_output_shape,
                               subsample=(2, 2), activation='linear',
                               border_mode='same', bias=False,
                               name='upscore2')(score_fr)
    # scale pool4 skip for compatibility
    scale_pool4 = Lambda(lambda x: x * 0.01, name='scale_pool4')(pool4)
    score_pool4 = Convolution2D(num_output, 1, 1, activation='linear',
                                border_mode='valid',
                                name='score_pool4')(scale_pool4)
    fuse_pool4 = merge([upscore2, score_pool4], mode='sum', name='fuse_pool4')
    _, fuse_pool4_dim_row, fuse_pool4_dim_col, _ = fuse_pool4.shape
    upscore_pool4_output_shape = (None,
                                  2 * int(fuse_pool4_dim_row),
                                  2 * int(fuse_pool4_dim_col),
                                  num_output)
    upscore_pool4 = Deconvolution2D(num_output, 4, 4,
                                    output_shape=upscore_pool4_output_shape,
                                    subsample=(2, 2), activation='linear',
                                    border_mode='same', bias=False,
                                    name='upscore_pool4')(fuse_pool4)
    # scale pool3 skip for compatibility
    scale_pool3 = Lambda(lambda x: x * 0.0001, name='scale_pool3')(pool3)
    score_pool3 = Convolution2D(num_output, 1, 1, activation='linear',
                                border_mode='valid',
                                name='score_pool3')(scale_pool3)
    fuse_pool3 = merge([upscore_pool4, score_pool3], mode='sum',
                       name='fuse_pool3')
    # score
    _, input_dim_row, input_dim_col, _ = input.shape
    score_output_shape = (None,
                          int(input_dim_row),
                          int(input_dim_col),
                          num_output)
    score = Deconvolution2D(num_output, 16, 16,
                            output_shape=score_output_shape,
                            subsample=(8, 8), activation='linear',
                            border_mode='same', bias=False,
                            name='score')(fuse_pool3)

    model = Model(input, score, name='fcn8s')

    return model
