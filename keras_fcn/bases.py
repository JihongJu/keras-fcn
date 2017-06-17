from __future__ import (
    absolute_import,
    unicode_literals
)
import abc
from keras.layers import (
    Dropout
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    ZeroPadding2D
)


class BaseNet(object):
    """Abstract Base Net for FCN."""

    def __new__(cls, *args, **kwargs):
        """New method."""
        return super(BaseNet, cls).__new__(cls).__call__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Call method."""
        return self.build.__func__

    @abc.abstractmethod
    def build(input):
        """Build the basenet on top of input.

        Arguments:
            input: Tensor of inputs.
        Returns:
            skip_layers: A list of upsampling entries for the skip
            architecture.

        """
        return [input]


class VGG16(BaseNet):
    """VGG base net.

    Examples:
        skip_layers = VGG16()(Input(shape=(26, 26, 3)))

    """

    def __new__(cls, trainable=False):
        """New method."""
        # VGG16 weights with no top layers
        cls.WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        return super(VGG16, cls).__new__(cls)

    def build(input):
        """Return pool3, pool4 and drop7 of VGG16."""
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

        return [pool3, pool4, drop7]
