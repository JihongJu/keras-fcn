import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.callbacks import Callback


class CheckNumericsOps(Callback):
    """Callback that terminates training when a NaN loss is encountered."""

    def __init__(self, validation_data, batch_size=1, histogram_freq=1):
        super(CheckNumericsOps, self).__init__()
        self.check_num = None
        self.batch_size = batch_size
        self.histogram_freq = histogram_freq
        self.validation_data = validation_data


    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        self.check_num = tf.add_check_numerics_ops()


    def on_batch_end(self, batch, logs=None):
        if self.validation_data and self.histogram_freq:
            if batch % self.histogram_freq == 0:
                for layer in self.model.layers:
                    functor = K.function([self.model.input, K.learning_phase()], [layer.output])
                    layer_out = functor(self.validation_data)
                    if np.any(np.isnan(layer_out)) or np.any(np.isinf(layer_out)):
                        print('The output of {} becomes nan'.format(layer.name))
                        self.model.stop_training = True

