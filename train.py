import numpy as np

from fcn import FCN
from voc_generator import ImageSegmentationGenerator, ImageDataLoader

train_datagen = ImageSegmentationGenerator()
test_datagen = ImageSegmentationGenerator()

image_data_loader = ImageDataLoader(
        image_dir='data/VOC2011/JPEGImages',
        image_format='jpg',
        color_mode='rgb',
        target_size=(224, 224))
segmentation_data_loader = ImageDataLoader(
        image_dir='data/VOC2011/SegmentationClass',
        image_format='png',
        color_mode='grayscale',
        target_size=(224, 224))

train_generator = train_datagen.flow_from_imageset(
        image_set='data/VOC2011/ImageSets/Segmentation/trainval.txt',
        image_data_loader=image_data_loader,
        segmentation_data_loader=segmentation_data_loader,
        batch_size=1)
test_generator = test_datagen.flow_from_imageset(
        image_set='data/VOC2011/ImageSets/Segmentation/test.txt',
        image_data_loader=image_data_loader,
        segmentation_data_loader=segmentation_data_loader,
        batch_size=1)

fcn8s = FCN(basenet='vgg16', input_shape=(224, 224, 3), num_output=21)
fcn8s.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

fcn8s.fit_generator(train_generator,
                    samples_per_epoch=2223,
                    nb_epoch=10000,
                    validation_generator=test_generator)
