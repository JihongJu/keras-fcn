import keras.backend as K
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

def Mean_IoU(classes):
    def mean_iou(y_true, y_pred):
        mean_iou, op = tf.metrics.mean_iou(y_true, y_pred, classes)
        return mean_iou
    _initialize_variables()
    return mean_iou


def _initialize_variables():
    """Utility to initialize uninitialized variables on the fly.
    """
    variables = tf.local_variables()
    uninitialized_variables = []
    for v in variables:
        if not hasattr(v, '_keras_initialized') or not v._keras_initialized:
            uninitialized_variables.append(v)
            v._keras_initialized = True
    if uninitialized_variables:
        sess = K.get_session()
        sess.run(tf.variables_initializer(uninitialized_variables))
