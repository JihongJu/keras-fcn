import pytest
from fcn import FCN


@pytest.fixture
def is_same_shape():
    def f(shape_a, shape_b):
        for dim in shape_a:
            if dim and dim not in shape_b:
                return False
        return True
    return f


def test_fcn_vgg16(is_same_shape):
    input_shape = (500, 500, 3)
    fcn_vgg16 = FCN(input_shape=input_shape)
    for l in fcn_vgg16.layers:
        test_shape = (None, None)
        if l.name == 'pool1':
            test_shape = (349, 349, 64)
        elif l.name == 'pool2':
            test_shape = (175, 175, 128)
        elif l.name == 'pool3':
            test_shape = (88, 88, 256)
        elif l.name == 'pool4':
            test_shape = (44, 44, 512)
        elif l.name == 'pool5':
            test_shape = (22, 22, 512)
        elif l.name == 'drop7':
            test_shape = (16, 16, 4096)
        elif l.name == 'upscore2':
            test_shape = (34, 34, 21)
        elif l.name == 'upscore_pool4':
            test_shape = (70, 70, 21)
        elif l.name == 'upscore8':
            test_shape = (568, 568, 21)
        elif l.name == 'score':
            test_shape = (500, 500, 21)
        assert is_same_shape(test_shape, l.output_shape)
    assert is_same_shape((500, 500, 21), fcn_vgg16.output_shape)

    input_shape = (250, 250, 3)
    fcn_vgg16 = FCN(input_shape=input_shape)
    assert is_same_shape((250, 250, 21), fcn_vgg16.output_shape)

    input_shape = (1366, 768, 3)
    fcn_vgg16 = FCN(input_shape=input_shape)
    assert is_same_shape((1366, 768, 21), fcn_vgg16.output_shape)

    input_shape = (224, 224, 3)
    fcn_vgg16 = FCN(input_shape=input_shape)
    assert is_same_shape((224, 224, 21), fcn_vgg16.output_shape)

    input_shape = (112, 112, 3)
    fcn_vgg16 = FCN(input_shape=input_shape)
    assert is_same_shape((112, 112, 21), fcn_vgg16.output_shape)
