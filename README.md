# keras-fcn
A re-implementation of Fully Convolutional Networks with Keras

### Installation

This implementation is based on Keras together with the GPU version of Tensorflow. Therefore, it is highly recommended to use a container, e.g. [Docker](https://www.docker.com/), to run the training processes.

The following installation procedures assumes Nvidia driver, [docker](https://docs.docker.com/engine/installation/linux/ubuntu/) and [nvidia-docker](https://devblogs.nvidia.com/parallelforall/nvidia-docker-gpu-server-application-deployment-made-easy/) are properly installed on a Ubuntu machine.

First clone the repository:

```
$ git clone https://github.com/JihongJu/keras-fcn.git
```

Start bash in a docker image and mount the local repository to `/workspace`

```
$ nvidia-docker run -it --rm -v `pwd`/keras-fcn/:/workspace jihong/nvidia-keras bash
```

Install requirements

```
# pip install -r requirements.txt
```

Validate installation by by starting an example train process

```
/workspace# python train.py
```


### Usage

Import FCN8s model and compile

```
fcn8s = FCN(basenet='vgg16', input_shape=(224, 224, 3), num_output=21)
fcn8s.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

#### Prepare the train set

Create a pseudo dataset :

```python
import numpy as np
X_train = np.random.rand(10, 128, 128, 3)
labels = np.random.randint(0, 21, size=[10, 128, 128])
y_train = np.eye(21)[labels]
```
Or alternatively use a real dataset, e.g. PASCAL VOC2011:

```python
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
```

#### Train

For the pseudo dataset:

```python
from fcn import FCN
fcn8s = FCN(basenet='vgg16', input_shape=(128, 128, 3), num_output=21)
fcn8s.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
fcn8s.fit(X_train, y_train, batch_size=1)
```

For PASCAL VOC:

```python
from fcn import FCN
fcn8s = FCN(basenet='vgg16', input_shape=(224, 224, 3), num_output=21)
fcn8s.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
fcn8s.fit_generator(train_generator,
                    samples_per_epoch=2223,
                    nb_epoch=10000,
                    validation_generator=test_generator)
```
