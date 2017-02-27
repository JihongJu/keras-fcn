import pytest
from fcn import FCN


def test_FCN8s():
    input_shape = (128, 128, 3)
    fcn8s = FCN(input_shape=input_shape)
    _, h, w, c = fcn8s.output_shape
    assert (h, w, c) == (128, 128, 21)
    # input_shape = (224, 224, 3)
    # fcn8s = FCN8s(input_shape=input_shape)
    # _, h, w, c = fcn8s.output_shape
    # assert (h, w, c) == (224, 224, 21)
    # input_shape = (1366, 988, 3)
    # fcn8s = FCN8s(input_shape=input_shape)
    # _, h, w, c = fcn8s.output_shape
    # assert (h, w, c) == (500, 500, 21)
