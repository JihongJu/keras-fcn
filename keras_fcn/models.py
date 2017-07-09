"""Fully Convolutional Neural Networks."""
from __future__ import (
    absolute_import,
    unicode_literals
)
import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Flatten, Activation, Reshape


from keras_fcn.encoders import VGG16, VGG19
from keras_fcn.decoders import VGGDecoder, VGGUpsampler


def FCN(*args, **kwargs):
    """Fully Convolutional Networks for semantic segmentation with VGG16.

    # Arguments
        input_shape: input image shape
        classes: number of classes
        trainable_encoder: Bool whether the weights of encoder are trainable
        weights: pre-trained weights to load (None if training from scratch)
    # Returns
        A Keras model instance

    """
    return FCN_VGG16(*args, **kwargs)


def FCN_VGG16(input_shape, classes,
              trainable_encoder=True, weights=None):
    """Fully Convolutional Networks for semantic segmentation with VGG16.

    # Arguments
        input_shape: input image shape
        classes: number of classes
        trainable_encoder: Bool whether the weights of encoder are trainable
        weights: pre-trained weights to load (None for training from scratch)



    # Returns
        A Keras model instance

    """
    # input
    inputs = Input(shape=input_shape)

    # Get the feature pyramid [drop7, pool4, pool3] from the VGG16 encoder
    pyramid_layers = 3
    encoder = VGG16(inputs, weights=weights, trainable=trainable_encoder)
    feat_pyramid = encoder.outputs[:pyramid_layers]

    # Append image to the end of feature pyramid
    feat_pyramid.append(inputs)

    # Decode feature pyramid
    # outputs = VGGDecoder(feat_pyramid, scales=[1, 1e-2, 1e-4], classes=21)
    outputs = VGGUpsampler(feat_pyramid, scales=[1, 1e-2, 1e-4], classes=21)

    # Flatten
    # if K.image_data_format() == 'channels_last':
    #     scores = keras.activations.softmax(outputs, axis=-1)
    # else:
    #     scores = keras.activations.softmax(outputs, axis=1)
    #outputs = Reshape((-1, classes), name='flatten_score')(outputs)
    scores = Activation('softmax')(outputs)
    # return model
    return Model(inputs=inputs, outputs=scores)


def FCN_VGG19(input_shape, classes,
              trainable_encoder=True, weights='imagenet'):
    """Fully Convolutional Networks for semantic segmentation with VGG16.

    # Arguments
        input_shape: input image shape
        classes: number of classes
        trainable_encoder: Bool whether the weights of encoder are trainable
        weights: pre-trained weights to load (None for training from scratch)



    # Returns
        A Keras model instance

    """
    # input
    inputs = Input(shape=input_shape)

    # Get the feature pyramid [drop7, pool4, pool3] from the VGG16 encoder
    pyramid_layers = 3
    encoder = VGG19(inputs, weights='imagenet', trainable=trainable_encoder)
    feat_pyramid = encoder.outputs[:pyramid_layers]

    # Append image to the end of feature pyramid
    feat_pyramid.append(inputs)

    # Decode feature pyramid
    outputs = VGGDecoder(feat_pyramid, scales=[1, 1e-2, 1e-4], classes=21)

    # return model
    return Model(inputs=inputs, outputs=outputs)
