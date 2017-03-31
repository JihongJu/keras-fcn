import pytest
from fcn.fcn import FCN, BaseNet, _get_basenet
from keras import backend as K


@pytest.fixture
def is_same_shape():
    def f(shape_a, shape_b):
        for dim in shape_a:
            if dim and dim not in shape_b:
                return False
        return True
    return f


@pytest.fixture
def fcn_test():
    def f(input_shape, **kwargs):
        if K.image_data_format() == 'channels_first':
            input_shape = (input_shape[2], input_shape[0], input_shape[1])
        model = FCN(input_shape=input_shape, **kwargs)
        return model
    return f


def test_fcn_vgg16_compile(fcn_test, is_same_shape):
    for data_format in {'channels_first', 'channels_last'}:
        K.set_image_data_format(data_format)
        fcn_vgg16 = fcn_test(input_shape=(500, 500, 3))
        fcn_vgg16.compile(loss='categorical_crossentropy', optimizer='sgd')
        assert True, "Failded to compile with {}".format(K.image_data_format())


def test_fcn_vgg16_shape(fcn_test, is_same_shape):
    input_shape = (500, 500, 3)
    fcn_vgg16 = fcn_test(input_shape=input_shape)
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

    input_shape = (1366, 768, 3)
    fcn_vgg16 = fcn_test(input_shape=input_shape)
    assert is_same_shape((1366, 768, 21), fcn_vgg16.output_shape)

    input_shape = (224, 224, 3)
    fcn_vgg16 = fcn_test(input_shape=input_shape)
    assert is_same_shape((224, 224, 21), fcn_vgg16.output_shape)


def test_get_basenet(fcn_test):
    input_shape = (500, 500, 3)
    with pytest.raises(ValueError):
        fcn_vgg16 = fcn_test(input_shape, basenet='vgg')
    with pytest.raises(ValueError):
        fcn_vgg16 = fcn_test(input_shape, basenet=16)


@pytest.fixture
def basenet_test():
    def f():
        basenet_test = BaseNet()
        return basenet_test
    return f


def test_basenet(basenet_test):
    assert basenet_test()(3) == [3]
