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
    Deconvolution2D
)


def FCN8s(weights=None, num_output=21,
          input_shape=(None, None, 3)):
    """Instantiate the VGG16-FCN8s architecture with tensorflow and keras
    # Arguments
        weights: pre-trained models
        num_output: number of classes
        input_shape: input image shape (Support only 32, 64, 128, ... for
        now)
    # Returns
        A Keras model instance
    """
    # input
    img_input = Input(shape=input_shape)
    # the base net: VGG16
    # Block 1
    conv1_1 = Convolution2D(64, 3, 3, activation='relu',
                            border_mode='same', name='conv1_1')(img_input)
    conv1_2 = Convolution2D(64, 3, 3, activation='relu',
                            border_mode='same', name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1_2)
    # Block 2
    conv2_1 = Convolution2D(128, 3, 3, activation='relu',
                            border_mode='same', name='conv2_1')(pool1)
    conv2_2 = Convolution2D(128, 3, 3, activation='relu',
                            border_mode='same', name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2_2)
    # Block 3
    conv3_1 = Convolution2D(256, 3, 3, activation='relu',
                            border_mode='same', name='conv3_1')(pool2)
    conv3_2 = Convolution2D(256, 3, 3, activation='relu',
                            border_mode='same', name='conv3_2')(conv3_1)
    conv3_3 = Convolution2D(256, 3, 3, activation='relu',
                            border_mode='same', name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3_3)
    # Block 4
    conv4_1 = Convolution2D(512, 3, 3, activation='relu',
                            border_mode='same', name='conv4_1')(pool3)
    conv4_2 = Convolution2D(512, 3, 3, activation='relu',
                            border_mode='same', name='conv4_2')(conv4_1)
    conv4_3 = Convolution2D(512, 3, 3, activation='relu',
                            border_mode='same', name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4_3)
    # Block 5
    conv5_1 = Convolution2D(512, 3, 3, activation='relu',
                            border_mode='same', name='conv5_1')(pool4)
    conv5_2 = Convolution2D(512, 3, 3, activation='relu',
                            border_mode='same', name='conv5_2')(conv5_1)
    conv5_3 = Convolution2D(512, 3, 3, activation='relu',
                            border_mode='same', name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5_3)
    # fully conv
    fc6 = Convolution2D(4096, 7, 7, activation='relu',
                        border_mode='same', name='fc6')(pool5)
    drop6 = Dropout(0.5)(fc6)
    fc7 = Convolution2D(4096, 1, 1, activation='relu',
                        border_mode='valid', name='fc7')(drop6)
    drop7 = Dropout(0.5)(fc7)

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
    _, input_dim_row, input_dim_col, _ = img_input.shape
    score_output_shape = (None,
                          int(input_dim_row),
                          int(input_dim_col),
                          num_output)
    score = Deconvolution2D(num_output, 16, 16,
                            output_shape=score_output_shape,
                            subsample=(8, 8), activation='linear',
                            border_mode='same', bias=False,
                            name='score')(fuse_pool3)

    model = Model(img_input, score, name='fcn8s')

    return model
