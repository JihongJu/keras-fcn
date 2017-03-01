from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)
import six
import numpy as np
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Dropout,
    Flatten,
    Lambda,
    merge
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    Deconvolution2D,
    ZeroPadding2D,
    Cropping2D
)
from keras import backend as K


DEBUG = 0


def _deconv(**deconv_params):
    def f(input):
        """Helper to build Deconvolution2D layer
        """
        nb_filter = deconv_params["nb_filter"]
        nb_row = deconv_params["nb_row"]
        nb_col = deconv_params["nb_col"]
        subsample = deconv_params.setdefault("subsample", (2, 2))
        border_mode = deconv_params.setdefault("border_mode", "valid")
        bias = deconv_params.setdefault("bias", False)
        name = deconv_params.setdefault("name", "deconv")
        width = input._keras_shape[ROW_AXIS]
        height = input._keras_shape[COL_AXIS]
        if border_mode == 'valid':
            # No zero padding, non-unit strides, transposed input-output
            # shape relationship
            output_width, output_height = tuple(
                np.array(subsample) * (np.array([width, height]) - 1)
                + np.array([nb_row, nb_col]))
        elif border_mode == 'same':
            # TODO Half (same) zero padding, non-unit strides, transposed
            # input-output shape relationship
            output_width, output_height = tuple(
                np.array(subsample) * np.array([width, height]))
        else:
            assert "border_mode for _deconv layers should be valid or same"
        if K.image_dim_ordering() == 'tf':
            output_shape = (None, output_width, output_height, nb_filter)
        else:
            output_shape = (None, nb_filter, 2 * width, 2 * height)
        deconv = Deconvolution2D(nb_filter=nb_filter, nb_row=nb_row,
                                 nb_col=nb_col, output_shape=output_shape,
                                 border_mode=border_mode, bias=bias,
                                 subsample=subsample, name=name)(input)

        return deconv

    return f


def _crop_and_merge(target_layer, offset_row, offset_col, merge_mode=None,
                    name='crop_and_merge'):
    """Crop the uncropped bottom such that it can be merged to a target_layer
    with mode `sum`
    """
    def f(input):
        width = input._keras_shape[ROW_AXIS]
        height = input._keras_shape[COL_AXIS]
        target_width = target_layer._keras_shape[ROW_AXIS]
        target_height = target_layer._keras_shape[COL_AXIS]
        cropped = Cropping2D(cropping=((offset_row,
                                        width - offset_row - target_width),
                                       (offset_col,
                                        height - offset_row - target_height)),
                             name='{}c'.format(name))(input)
        if merge_mode is None:
            return cropped
        else:
            return merge([cropped, target_layer], mode=merge_mode,
                         name='fuse_{}'.format(name))

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
        pad1 = ZeroPadding2D(padding=(100, 100))(input)
        conv1_1 = Convolution2D(64, 3, 3, activation='relu',
                                border_mode='valid',
                                name='conv1_1')(pad1)
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
        pool2 = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                             name='block2_pool')(conv2_2)
        # Block 3
        conv3_1 = Convolution2D(256, 3, 3, activation='relu',
                                border_mode='same', name='conv3_1')(pool2)
        conv3_2 = Convolution2D(256, 3, 3, activation='relu',
                                border_mode='same', name='conv3_2')(conv3_1)
        conv3_3 = Convolution2D(256, 3, 3, activation='relu',
                                border_mode='same', name='conv3_3')(conv3_2)
        pool3 = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                             name='block3_pool')(conv3_3)
        # Block 4
        conv4_1 = Convolution2D(512, 3, 3, activation='relu',
                                border_mode='same', name='conv4_1')(pool3)
        conv4_2 = Convolution2D(512, 3, 3, activation='relu',
                                border_mode='same', name='conv4_2')(conv4_1)
        conv4_3 = Convolution2D(512, 3, 3, activation='relu',
                                border_mode='same', name='conv4_3')(conv4_2)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                             name='block4_pool')(conv4_3)
        # Block 5
        conv5_1 = Convolution2D(512, 3, 3, activation='relu',
                                border_mode='same', name='conv5_1')(pool4)
        conv5_2 = Convolution2D(512, 3, 3, activation='relu',
                                border_mode='same', name='conv5_2')(conv5_1)
        conv5_3 = Convolution2D(512, 3, 3, activation='relu',
                                border_mode='same', name='conv5_3')(conv5_2)
        pool5 = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                             name='block5_pool')(conv5_3)
        # fully conv
        if skip:
            fc6 = Convolution2D(4096, 7, 7, activation='relu',
                                border_mode='valid', name='fc6')(pool5)
            drop6 = Dropout(0.5)(fc6)
            fc7 = Convolution2D(4096, 1, 1, activation='relu',
                                border_mode='valid', name='fc7')(drop6)
            drop7 = Dropout(0.5)(fc7)
            return (pool3, pool4, drop7)
        # classification
        else:
            pool5f = Flatten(name='pool5f')(pool5)
            fc6 = Dense(4096, activation='relu', name='fc6')(pool5f)
            fc7 = Dense(4096, activation='relu', name='fc7')(fc6)
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
    # Get pool3, pool4 and drop7 from the base net: VGG16
    pool3, pool4, drop7 = basenet(skip=True)(input)

    score_fr = Convolution2D(num_output, 1, 1, activation='linear',
                             border_mode='valid', name='score_fr')(drop7)
    upscore2 = _deconv(nb_filter=num_output, nb_row=4, nb_col=4,
                       subsample=(2, 2), border_mode='valid',
                       name='upscore2')(score_fr)
    # scale pool4 skip for compatibility
    scale_pool4 = Lambda(lambda x: x * 0.01, name='scale_pool4')(pool4)
    score_pool4 = Convolution2D(num_output, 1, 1, activation='linear',
                                border_mode='valid',
                                name='score_pool4')(scale_pool4)
    fuse_pool4 = _crop_and_merge(upscore2, offset_row=5, offset_col=5,
                                 merge_mode='sum',
                                 name='score_pool4')(score_pool4)
    upscore_pool4 = _deconv(nb_filter=num_output, nb_row=4, nb_col=4,
                            subsample=(2, 2), border_mode='valid',
                            name='upscore_pool4')(fuse_pool4)
    # scale pool3 skip for compatibility
    scale_pool3 = Lambda(lambda x: x * 0.0001, name='scale_pool3')(pool3)
    score_pool3 = Convolution2D(num_output, 1, 1, activation='linear',
                                border_mode='valid',
                                name='score_pool3')(scale_pool3)
    fuse_pool3 = _crop_and_merge(upscore_pool4, offset_row=9, offset_col=9,
                                 merge_mode='sum',
                                 name='score_pool3')(score_pool3)
    # score
    upscore8 = _deconv(nb_filter=num_output, nb_row=16, nb_col=16,
                       subsample=(8, 8), border_mode='valid',
                       name='upscore8')(fuse_pool3)
    score = _crop_and_merge(input, offset_row=31, offset_col=31,
                            merge_mode=None,
                            name='score')(upscore8)
    # model
    model = Model(input, score, name='fcn8s')
    if DEBUG:
        for l in model.layers:
            print("Layers output shape of the model:")
            print("{} has shape: {}".format(l.name, l.output_shape))

    return model
