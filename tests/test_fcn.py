import pytest
from fcn import FCN
from keras.utils.visualize_util import plot


def test_FCN8s():
    input_shape = (500, 500, 3)
    fcn8s = FCN(input_shape=input_shape)
    _, h, w, c = fcn8s.output_shape
    plot(fcn8s, to_file='fcn8s.png')
    assert (h, w, c) == (500, 500, 21)
    input_shape = (250, 250, 3)
    fcn8s = FCN(input_shape=input_shape)
    _, h, w, c = fcn8s.output_shape
    assert (h, w, c) == (250, 250, 21)
    input_shape = (1366, 768, 3)
    fcn8s = FCN(input_shape=input_shape)
    _, h, w, c = fcn8s.output_shape
    assert (h, w, c) == (1366, 768, 21)
    input_shape = (224, 224, 3)
    fcn8s = FCN(input_shape=input_shape)
    _, h, w, c = fcn8s.output_shape
    assert (h, w, c) == (224, 224, 21)
    input_shape = (112, 112, 3)
    fcn8s = FCN(input_shape=input_shape)
    _, h, w, c = fcn8s.output_shape
    assert (h, w, c) == (112, 112, 21)
    input_shape = (32, 32, 3)
    fcn8s = FCN(input_shape=input_shape)
    _, h, w, c = fcn8s.output_shape
    assert (h, w, c) == (32, 32, 21)
