import pytest
from voc_generator import ImageSegmentationGenerator, ImageDataLoader


@pytest.fixture
def voc_data_gen():
    voc_datagen = ImageSegmentationGenerator()
    return voc_datagen


@pytest.fixture
def image_data_loader():
    data_load_args = dict(
        image_dir='data/VOC2011/JPEGImages',
        image_format='jpg',
        color_mode='rgb',
        target_size=(500, 500),
        save_to_dir='tests',
        save_prefix='resized_image_'
    )
    return ImageDataLoader(**data_load_args)


@pytest.fixture
def segmentation_data_loader():
    data_load_args = dict(
        image_dir='data/VOC2011/SegmentationClass',
        image_format='png',
        color_mode='grayscale',
        target_size=(500, 500),
        save_to_dir='tests',
        save_prefix='resized_segmentation_'
    )
    return ImageDataLoader(**data_load_args)


def test_flow_from_imageset(voc_data_gen, image_data_loader,
                            segmentation_data_loader):
    train_generator = voc_data_gen.flow_from_imageset(
        image_set='data/VOC2011/ImageSets/Segmentation/trainval.txt',
        image_data_loader=image_data_loader,
        segmentation_data_loader=segmentation_data_loader,
        batch_size=1)
    batch_x, batch_y = train_generator.next()
    print(batch_x.shape, batch_y.shape)
    assert False
