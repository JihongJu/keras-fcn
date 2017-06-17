"""Fully Convolutional Neural Networks."""
from __future__ import (
    absolute_import,
    unicode_literals
)

from keras.models import Model
from keras.layers import (
    Input,
    Lambda
)
from keras.layers.convolutional import (
    Conv2D,
    Conv2DTranspose
)

from keras.layers.merge import add
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import layer_utils
from keras_fcn.bases import VGG16
from keras_fcn.layers import CroppingLike2D
from keras_fcn.backend import load_weights


def FCN(basenet='vgg16', trainable_base=False,
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
    # input
    input = Input(shape=input_shape)
    # Get feat_pyramids=[pool3, pool4, drop7] from the base net: VGG16
    feat_pyramids = VGG16()(input)

    drop7 = feat_pyramids[2]
    score_fr = Conv2D(filters=num_output, kernel_size=(1, 1),
                      padding='valid',
                      name='score_fr')(drop7)
    upscore2 = Conv2DTranspose(filters=num_output, kernel_size=(4, 4),
                               strides=(2, 2), padding='valid', use_bias=False,
                               name='upscore2')(score_fr)

    # scale pool4 skip for compatibility
    pool4 = feat_pyramids[1]
    scale_pool4 = Lambda(lambda x: x * 0.01, name='scale_pool4')(pool4)
    score_pool4 = Conv2D(filters=num_output, kernel_size=(1, 1),
                         padding='valid', name='score_pool4')(scale_pool4)
    score_pool4c = CroppingLike2D(upscore2, offset=(5, 5),
                                  name='score_pool4c')(score_pool4)
    fuse_pool4 = add([upscore2, score_pool4c])
    upscore_pool4 = Conv2DTranspose(filters=num_output, kernel_size=(4, 4),
                                    strides=(2, 2), padding='valid',
                                    use_bias=False,
                                    name='upscore_pool4')(fuse_pool4)
    # scale pool3 skip for compatibility
    pool3 = feat_pyramids[0]
    scale_pool3 = Lambda(lambda x: x * 0.0001, name='scale_pool3')(pool3)
    score_pool3 = Conv2D(filters=num_output, kernel_size=(1, 1),
                         padding='valid', name='score_pool3')(scale_pool3)
    score_pool3c = CroppingLike2D(upscore_pool4, offset=(9, 9),
                                  name='score_pool3c')(score_pool3)
    fuse_pool3 = add([upscore_pool4, score_pool3c])

    # score
    upscore8 = Conv2DTranspose(filters=num_output, kernel_size=(16, 16),
                               strides=(8, 8), padding='valid',
                               use_bias=False,
                               name='upscore8')(fuse_pool3)
    score = CroppingLike2D(input, offset=(31, 31), name='score')(upscore8)

    # model
    model = Model(input, score, name='fcn_vgg16')

    # load weights
    if weights == 'imagenet':
        weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                VGG16.WEIGHTS_PATH,
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
