# keras-fcn

[![Build Status](https://travis-ci.org/JihongJu/keras-fcn.svg?branch=master)](https://travis-ci.org/JihongJu/keras-fcn) [![codecov](https://codecov.io/gh/jihongju/keras-fcn/branch/master/graph/badge.svg)](https://codecov.io/gh/jihongju/keras-fcn)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A re-implementation of Fully Convolutional Networks with Keras

## Installation

### Dependencies


1. [keras](https://keras.io/#installation)
2. [tensorflow](https://www.tensorflow.org/install/)/[theano](http://deeplearning.net/software/theano/install.html)/[CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine) (CNTK is not tested.)


### Install with `pip`

```bash
$ pip install git+https://github.com/JihongJu/keras-fcn.git
```


### Build from source

```bash
$ git clone https://github.com/JihongJu/keras-fcn.git
$ cd keras-fcn
$ pip install --editable .
```

## Usage

### FCN with VGG16

```python
from keras_fcn import FCN
fcn_vgg16 = FCN(input_shape=(500, 500, 3), classes=21,  
                weights='imagenet', trainable_encoder=True)
fcn_vgg16.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
fcn_vgg16.fit(X_train, y_train, batch_size=1)
```

### FCN with VGG19

```python
from keras_fcn import FCN
fcn_vgg19 = FCN_VGG19(input_shape=(500, 500, 3), classes=21,  
                      weights='imagenet', trainable_encoder=True)
fcn_vgg19.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
fcn_vgg19.fit(X_train, y_train, batch_size=1)
```

### Custom FCN (VGG16 as an example)

```python
from keras.layers import Input
from keras.models import Model
from keras_fcn.encoders import Encoder
from keras_fcn.decoders import VGGDecoder
from keras_fcn.blocks import (vgg_conv, vgg_fc)
inputs = Input(shape=(224, 224, 3))
blocks = [vgg_conv(64, 2, 'block1'),
          vgg_conv(128, 2, 'block2'),
          vgg_conv(256, 3, 'block3'),
          vgg_conv(512, 3, 'block4'),
          vgg_conv(512, 3, 'block5'),
          vgg_fc(4096)]
encoder = Encoder(inputs, blocks, weights='imagenet',
                  trainable=True)
feat_pyramid = encoder.outputs   # A feature pyramid with 5 scales
feat_pyramid = feat_pyramid[:3]  # Select only the top three scale of the pyramid
feat_pyramid.append(inputs)      # Add image to the bottom of the pyramid


outputs = VGGDecoder(feat_pyramid, scales=[1, 1e-2, 1e-4], classes=21)

fcn_custom = Model(inputs=inputs, outputs=outputs)
```

And implement a custom Fully Convolutional Network becomes simply define a series of convolutional blocks that one stacks on top of another.

### Custom decoders

```python
from keras_fcn.blocks import vgg_deconv, vgg_score
from keras_fcn.decoders import Decoder
decode_blocks = [
vgg_deconv(classes=21, scale=1),            
vgg_deconv(classes=21, scale=0.01),
vgg_deconv(classes=21, scale=0.0001, kernel_size=(16,16), strides=(8,8)),
vgg_score(crop_offset='centered')           # A functional block cropping the
                                            # outcome scores to match the image.
                                            # Can use together with other custom
                                            # blocks
]
outputs = Decoder(feat_pyramid, decode_blocks)

```

The `decode_blocks` can be customized as well.

```python
from keras_fcn.layers import CroppingLike2D
def my_decode_block(classes, scale):
    """A functional decoder block.
    :param: classes: Integer, number of classes
    :param scale: Float, weight of the current pyramid scale, varing from 0 to 1

    :return f: A function that takes a feature from the feature pyramid, x,
               applies upsampling and accumulate the result from the top of
               the pyramid.
    """
  def f(x, y):
    x = Lambda(lambda xx: xx * scale)(x)  # First weighs the scale
    x = Conv2D(filters=classes, kernel_size=(1,1))(x)   # Stride 1 conv layers,  
                                                        # replacing the
                                                        # traditional FC layer.
    if y is None:   # First block has no y.
      y = Conv2DTranspose(filters=classes, ...) # Deconvolutional layer or
                                                # Upsampling Layer
    else:
      x = CroppingLike2D(target=y, offset='centered')(x) # Crop the upsampled
                                                         # feature to match
                                                         # the output of one
                                                         # scale up.
      y = add([y, x])
      y = Conv2DTranspose(filters=classes, ...) # Deconv/Upsampling again.
    return y  # return output of the current scale in the feature pyramid
  return x

```

## Try Examples (The example is out-of-date for now)

1. Download [VOC2011](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/) dataset

```bash
$ wget "http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar"
$ tar -xvzf VOCtrainval_25-May-2011.tar
$ mkdir ~/Datasets
$ mv TrainVal/VOCdevkit/VOC2011 ~/Datasets
```

2. Mount dataset from host to container and start bash in container image

From repository `keras-fcn`

```bash
$ nvidia-docker run -it --rm -v `pwd`:/root/workspace -v ${Home}/Datasets/:/root/workspace/data jihong/keras-gpu bash
```

or equivalently,
```bash
$ make bash
```

3. Within the container, run the following codes.

```bash
$ cd ~/workspace
$ source venv/bin/activate
$ pip install -r requirements
$ pip setup.py build
$ cd voc2011
$ python train.py
```

More details see source code of the example in [Training Pascal VOC2011 Segmention](https://github.com/JihongJu/keras-fcn/blob/master/voc2011/train.py)


### Model Architecture

FCN8s with VGG16 as base net:

![fcn_vgg16](fcn_vgg16.png)


### TODO

 - Add ResNet
