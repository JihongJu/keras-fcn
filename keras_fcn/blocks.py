import keras.backend as K
from keras.layers import (
    Dropout,
    Lambda
)
from keras.layers.convolutional import (
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    ZeroPadding2D
)
from keras.layers.merge import add
from keras.regularizers import l2
from keras_fcn.layers import CroppingLike2D, BilinearUpSampling2D


def vgg_conv(filters, convs, padding=False, weight_decay=0., block_name='blockx'):
    """A VGG convolutional block for encoding.

    :param filters: Integer, number of filters per conv layer
    :param convs: Integer, number of conv layers in the block
    :param block_name: String, the name of the block, e.g., block1

    >>> from keras_fcn.blocks import vgg_conv
    >>> x = vgg_conv(filters=64, convs=2, block_name='block1')(x)

    """
    def f(x):
        for i in range(convs):
            if block_name == 'block1' and i == 0:
                if padding is True:
                    x = ZeroPadding2D(padding=(100, 100))(x)
                x = Conv2D(filters, (3, 3), activation='relu', padding='same',
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(weight_decay),
                           name='{}_conv{}'.format(block_name, int(i + 1)))(x)
            else:
                x = Conv2D(filters, (3, 3), activation='relu', padding='same',
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(weight_decay),
                           name='{}_conv{}'.format(block_name, int(i + 1)))(x)

        pool = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                            name='{}_pool'.format(block_name))(x)
        return pool
    return f


def vgg_fc(filters, weight_decay=0., block_name='block5'):
    """A fully convolutional block for encoding.

    :param filters: Integer, number of filters per fc layer

    >>> from keras_fcn.blocks import vgg_fc
    >>> x = vgg_fc(filters=4096)(x)

    """
    def f(x):
        fc6 = Conv2D(filters=4096, kernel_size=(7, 7),
                     activation='relu', padding='same',
                     dilation_rate=(2, 2),
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(weight_decay),
                     name='{}_fc6'.format(block_name))(x)
        drop6 = Dropout(0.5)(fc6)
        fc7 = Conv2D(filters=4096, kernel_size=(1, 1),
                     activation='relu', padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(weight_decay),
                     name='{}_fc7'.format(block_name))(drop6)
        drop7 = Dropout(0.5)(fc7)
        return drop7
    return f


def vgg_deconv(classes, scale=1, kernel_size=(4, 4), strides=(2, 2),
               crop_offset='centered', weight_decay=0., block_name='featx'):
    """A VGG convolutional transpose block for decoding.

    :param classes: Integer, number of classes
    :param scale: Float, scale factor to the input feature, varing from 0 to 1
    :param kernel_size: Tuple, the kernel size for Conv2DTranspose layers
    :param strides: Tuple, the strides for Conv2DTranspose layers
    :param crop_offset: Tuple or "centered", the offset for cropping.
    The default is "centered", which crop the center of the feature map.

    >>> from keras_fcn.blocks import vgg_deconv
    >>> x = vgg_deconv(classes=21, scale=1e-2, block_name='feat2')(x)

    """
    def f(x, y):
        def scaling(xx, ss=1):
            return xx * ss
        scaled = Lambda(scaling, arguments={'ss': scale},
                        name='scale_{}'.format(block_name))(x)
        score = Conv2D(filters=classes, kernel_size=(1, 1),
                       activation='linear',
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(weight_decay),
                       name='score_{}'.format(block_name))(scaled)
        if y is None:
            upscore = Conv2DTranspose(filters=classes, kernel_size=kernel_size,
                                      strides=strides, padding='valid',
                                      kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay),
                                      use_bias=False,
                                      name='upscore_{}'.format(block_name))(score)
        else:
            crop = CroppingLike2D(target_shape=K.int_shape(y),
                                  offset=crop_offset,
                                  name='crop_{}'.format(block_name))(score)
            merge = add([y, crop])
            upscore = Conv2DTranspose(filters=classes, kernel_size=kernel_size,
                                      strides=strides, padding='valid',
                                      kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay),
                                      use_bias=False,
                                      name='upscore_{}'.format(block_name))(merge)
        return upscore
    return f


def vgg_upsampling(classes, target_shape=None, scale=1, weight_decay=0., block_name='featx'):
    """A VGG convolutional block with bilinear upsampling for decoding.

    :param classes: Integer, number of classes
    :param scale: Float, scale factor to the input feature, varing from 0 to 1
    :param target_shape: 4D Tuples with targe_height, target_width as
    the 2nd, 3rd elements if `channels_last` or as the 3rd, 4th elements if
    `channels_first`.

    >>> from keras_fcn.blocks import vgg_upsampling
    >>> feat1, feat2, feat3 = feat_pyramid[:3]
    >>> y = vgg_upsampling(classes=21, target_shape=(None, 14, 14, None),
    >>>                    scale=1, block_name='feat1')(feat1, None)
    >>> y = vgg_upsampling(classes=21, target_shape=(None, 28, 28, None),
    >>>                    scale=1e-2, block_name='feat2')(feat2, y)
    >>> y = vgg_upsampling(classes=21, target_shape=(None, 224, 224, None),
    >>>                    scale=1e-4, block_name='feat3')(feat3, y)

    """
    def f(x, y):
        score = Conv2D(filters=classes, kernel_size=(1, 1),
                       activation='linear',
                       padding='valid',
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(weight_decay),
                       name='score_{}'.format(block_name))(x)
        if y is not None:
            def scaling(xx, ss=1):
                return xx * ss
            scaled = Lambda(scaling, arguments={'ss': scale},
                            name='scale_{}'.format(block_name))(score)
            score = add([y, scaled])
        upscore = BilinearUpSampling2D(
            target_shape=target_shape,
            name='upscore_{}'.format(block_name))(score)
        return upscore
    return f


def vgg_score(crop_offset='centered'):
    """A helper block to crop the decoded feature.

    :param crop_offset: Tuple or "centered", the offset for cropping.
    The default is "centered", which crop the center of the feature map.

    >>> from keras_fcn.blocks import vgg_deconv
    >>> score = vgg_score(crop_offset='centered')(image, upscore)

    """
    def f(x, y):
        score = CroppingLike2D(target_shape=K.int_shape(
            x), offset=crop_offset, name='score')(y)
        return score
    return f
