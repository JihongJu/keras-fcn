from __future__ import unicode_literals
import json
import pytest
import numpy as np
from voc2011.voc_generator import PascalVocGenerator, ImageSetLoader
from voc2011.utils import increment_var


import yaml
with open("init_args.yml", 'r') as stream:
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
    datagen = PascalVocGenerator(image_shape=[224, 224, 3],
                                 image_resample=True,
                                 pixelwise_center=True,
                                 pixel_mean=[104.00699, 116.66877, 122.67892])
    return datagen


@pytest.fixture
def voc_loader():
    return ImageSetLoader(**init_args['image_set_loader']['train'])


@pytest.fixture
def val_loader():
    return ImageSetLoader(**init_args['image_set_loader']['val'])

def test_flow_from_imageset(voc_datagen, voc_loader, is_same_shape):
    train_generator = voc_datagen.flow_from_imageset(class_mode='categorical',
                                                     classes=21,
                                                     batch_size=1,
                                                     shuffle=True,
                                                     image_set_loader=voc_loader)
    for _ in range(1111):
        batch_x, batch_y = train_generator.next()
        assert is_same_shape((1, 224, 224, 3), batch_x.shape)
        assert is_same_shape((1, 224, 224, 21), batch_y.shape)
        assert not np.all(batch_y == 0.)


#def test_loader(voc_datagen, voc_loader):
#    for fn in voc_loader.filenames:
#        x = voc_loader.load_img(fn)
#        x = voc_datagen.standardize(x)
#        y_true = voc_loader.load_seg(fn)
#        voc_loader.save(x, y_true, fn)
