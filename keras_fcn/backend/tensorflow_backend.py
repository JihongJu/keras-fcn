import numpy as np
import tensorflow as tf
import keras.backend as K


def resize_images(x, size, method='bilinear'):
    new_size = tf.convert_to_tensor(size, dtype=tf.int32)
    resized = tf.image.resize_images(x, new_size)
    return resized
