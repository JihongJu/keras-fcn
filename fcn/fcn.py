"""Fully Convolutional Neural Networks."""
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)
import abc
import six
import h5py
import numpy as np
from keras.models import Model
from keras.layers import (
    Input,
    Dropout,
    Lambda
)
from keras.layers.convolutional import (
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    ZeroPadding2D,
    Cropping2D
)
from keras.layers.merge import add
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import layer_utils

DEBUG = 0


def load_weights(model, weights_path):
    """Load weights from Caffe models."""
    # weights_data = np.load(weights_path).item()
    if h5py is None:
        raise ImportError('`load_weights` requires h5py.')
    f = h5py.File(weights_path, mode='r')

    # New file format.
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in model.layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]

        for layer in index.get(name, []):
            symbolic_weights = layer.weights
            # Set values.
            for i in range(len(weight_values)):
                weight_value_tuples.append((symbolic_weights[i],
                                            weight_values[i]))
    K.batch_set_value(weight_value_tuples)

    return layer_names


def _crop(target_layer, offset=(None, None), name=None):
    """Crop the bottom such that it has the same shape as target_layer."""
    def f(input):
        width = input._keras_shape[ROW_AXIS]
        height = input._keras_shape[COL_AXIS]
        target_width = target_layer._keras_shape[ROW_AXIS]
        target_height = target_layer._keras_shape[COL_AXIS]
        cropped = Cropping2D(cropping=((offset[0],
                                        width - offset[0] - target_width),
                                       (offset[1],
                                        height - offset[1] - target_height)),
                             name='{}'.format(name))(input)
        return cropped
    return f


class BaseNet(object):
    """Abstract BaseNet for FCN."""

    def __new__(cls, *args, **kwargs):
        """Init method."""
        return super(BaseNet, cls).__new__(cls).__call__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Call method."""
        return self._build.__func__

    @abc.abstractmethod
    def _build(input):
        """Build method."""
        """Build the basenet on top of input.

        Arguments:
            input: Tensor of inputs.
        Returns:
            skip_layers: A list of upsampling entries for the skip
                          architecture."""
        return [input]


class VGG16(BaseNet):
    """VGG base net.

    Examples:
        skip_layers = VGG16()(Input(shape=(26, 26, 3)))

    """

    # VGG16 weights with no top layers
    WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

    def _build(input):
        pad1 = ZeroPadding2D(padding=(100, 100))(input)
        conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                         padding='valid', name='block1_conv1')(pad1)
        conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                         padding='same', name='block1_conv2')(conv1_1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                             padding='same', name='pool1')(conv1_2)
        # Block 2
        conv2_1 = Conv2D(filters=128, kernel_size=(3, 3),
                         activation='relu',
                         padding='same', name='block2_conv1')(pool1)
        conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                         padding='same', name='block2_conv2')(conv2_1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                             padding='same', name='pool2')(conv2_2)
        # Block 3
        conv3_1 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                         padding='same', name='block3_conv1')(pool2)
        conv3_2 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                         padding='same', name='block3_conv2')(conv3_1)
        conv3_3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                         padding='same', name='block3_conv3')(conv3_2)
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                             padding='same', name='pool3')(conv3_3)
        # Block 4
        conv4_1 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                         padding='same', name='block4_conv1')(pool3)
        conv4_2 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                         padding='same', name='block4_conv2')(conv4_1)
        conv4_3 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                         padding='same', name='block4_conv3')(conv4_2)
        pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                             padding='same', name='pool4')(conv4_3)
        # Block 5
        conv5_1 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                         padding='same', name='block5_conv1')(pool4)
        conv5_2 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                         padding='same', name='block5_conv2')(conv5_1)
        conv5_3 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                         padding='same', name='block5_conv3')(conv5_2)
        pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                             padding='same', name='pool5')(conv5_3)
        # fully conv
        fc6 = Conv2D(filters=4096, kernel_size=(7, 7),
                     activation='relu', padding='valid',
                     name='fc6')(pool5)
        drop6 = Dropout(0.5)(fc6)
        fc7 = Conv2D(filters=4096, kernel_size=(1, 1),
                     activation='relu', padding='valid',
                     name='fc7')(drop6)
        drop7 = Dropout(0.5)(fc7)
        return [drop7, pool4, pool3]


def _get_basenet(identifier):
    """Get basenet by identifier."""
    if isinstance(identifier, six.string_types):
        basenet = globals().get(identifier.upper())
        if not basenet:
            raise ValueError('Invalid {}'.format(identifier))
        return basenet
    else:
        raise ValueError('Invalid {}. A string expected.'.format(identifier))


def _handle_data_format():
    """Image data format handler."""
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def FCN(basenet='vgg16', trainable_base=True,
        num_output=21, input_shape=(None, None, 3),
        weights='imagenet'):
    """Instantiate the FCN8s architecture with keras.

    # Arguments
        basenet: type of basene {'vgg16'}
        trainable_base: Bool whether the basenet weights are trainable
        num_output: number of classes
        input_shape: input image shape
        weights: pre-trained weights to load (None for training from scratch)
    # Returns
        A Keras model instance
    """
    _handle_data_format()
    basenet = _get_basenet(basenet)
    # input
    input = Input(shape=input_shape)
    # Get skip_layers=[drop7, pool4, pool3] from the base net: VGG16
    skip_layers = basenet(skip_architecture=True)(input)

    drop7 = skip_layers[0]
    score_fr = Conv2D(filters=num_output, kernel_size=(1, 1),
                      padding='valid',
                      name='score_fr')(drop7)
    upscore2 = Conv2DTranspose(filters=num_output, kernel_size=(4, 4),
                               strides=(2, 2), padding='valid', use_bias=False,
                               data_format=K.image_data_format(),
                               name='upscore2')(score_fr)
    # scale pool4 skip for compatibility
    pool4 = skip_layers[1]
    scale_pool4 = Lambda(lambda x: x * 0.01, name='scale_pool4')(pool4)
    score_pool4 = Conv2D(filters=num_output, kernel_size=(1, 1),
                         padding='valid', name='score_pool4')(scale_pool4)
    score_pool4c = _crop(upscore2, offset=(5, 5),
                         name='score_pool4c')(score_pool4)
    fuse_pool4 = add([upscore2, score_pool4c])
    upscore_pool4 = Conv2DTranspose(filters=num_output, kernel_size=(4, 4),
                                    strides=(2, 2), padding='valid',
                                    use_bias=False,
                                    data_format=K.image_data_format(),
                                    name='upscore_pool4')(fuse_pool4)
    # scale pool3 skip for compatibility
    pool3 = skip_layers[2]
    scale_pool3 = Lambda(lambda x: x * 0.0001, name='scale_pool3')(pool3)
    score_pool3 = Conv2D(filters=num_output, kernel_size=(1, 1),
                         padding='valid', name='score_pool3')(scale_pool3)
    score_pool3c = _crop(upscore_pool4, offset=(9, 9),
                         name='score_pool3c')(score_pool3)
    fuse_pool3 = add([upscore_pool4, score_pool3c])
    # score
    upscore8 = Conv2DTranspose(filters=num_output, kernel_size=(16, 16),
                               strides=(8, 8), padding='valid',
                               use_bias=False,
                               data_format=K.image_data_format(),
                               name='upscore8')(fuse_pool3)
    score = _crop(input, offset=(31, 31), name='score')(upscore8)

    # model
    model = Model(input, score, name='fcn_vgg16')

    # load weights
    if weights == 'imagenet':
        weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                basenet.WEIGHTS_PATH,
                                cache_subdir='models')
        layer_names = load_weights(model, weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)
        # Freezing basenet weights
        if not trainable_base:
            for layer in model.layers:
                if layer.name in layer_names:
                    layer.trainable = False

    return model
