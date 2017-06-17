import pytest
from voc2011.voc_generator import PascalVocGenerator, ImageSetLoader


import yaml
with open("tests/init_args.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)


@pytest.fixture
def is_same_shape():
    def f(shape_a, shape_b):
        for dim in shape_a:
            if dim and dim not in shape_b:
                return False
        return True
    return f


@pytest.fixture
def voc_datagen():
    voc_datagen = PascalVocGenerator(**init_args['pascal_voc_generator'][
        'train'])
    return voc_datagen


@pytest.fixture
def voc_loader():
    return ImageSetLoader(**init_args['image_set_loader']['train'])


def test_flow_from_imageset(voc_datagen, voc_loader, is_same_shape):
    flow_args = init_args['pascal_voc_generator']['flow_from_imageset']
    flow_args['image_set_loader'] = voc_loader
    for _ in range(2):
        train_generator = voc_datagen.flow_from_imageset(**flow_args)
        batch_x, batch_y = train_generator.next()
        assert is_same_shape((500, 500, 3), batch_x.shape)
        assert is_same_shape((500, 500, 21), batch_y.shape)
