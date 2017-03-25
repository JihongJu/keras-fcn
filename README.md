# keras-fcn

[![Build Status](https://travis-ci.org/JihongJu/keras-fcn.svg?branch=master)](https://travis-ci.org/JihongJu/keras-fcn)

[![codecov](https://codecov.io/gh/jihongju/keras-fcn/branch/master/graph/badge.svg)](https://codecov.io/gh/jihongju/keras-fcn)

A re-implementation of Fully Convolutional Networks with Keras

### Installation

This implementation is based on Keras and the GPU version of Tensorflow/Theano. It is highly recommended to use [Docker container](https://www.docker.com/), to run the training process.

0. Dependencies

The following installation procedures assumes Nvidia driver, [docker](https://docs.docker.com/engine/installation/linux/ubuntu/) and [nvidia-docker](https://devblogs.nvidia.com/parallelforall/nvidia-docker-gpu-server-application-deployment-made-easy/) are properly installed on an Ubuntu system.

1. Clone the repository:

```
$ git clone https://github.com/JihongJu/keras-fcn.git
```

2. Start bash in a docker image and mount the local repository to `/workspace`

```bash
$ nvidia-docker run -it --rm -v `pwd`/keras-fcn/:/root/workspace jihong/keras-gpu bash
```

3. Install requirements in the container

```bash
~# cd workspace
~/workspace# virtualenv --system-site-packages venv
~/workspace# source venv/bin/activate
~/workspace# pip install -r requirements.txt
```

Validate installation.

```
~/workspace# py.test tests/test_fcn.py
```

4. Quit container

`Ctrl+D` will do the job.


### Usage

Import FCN8s model and compile

```
from fcn import FCN
fcn_vgg16 = FCN(basenet='vgg16', input_shape=(500, 500, 3), num_output=21)
fcn_vgg16.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
fcn_vgg16.fit(X_train, y_train, batch_size=1)
```

More details see [Train Pascal VOC2011](https://github.com/JihongJu/keras-fcn/blob/develop/voc2011/train.py)

#### Prepare data

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
$ ./docker_bash.sh
```

#### Train

Train with VOC2011

```bash
~# cd workspace/voc2011
~/workspace/voc2011# python train.py
```

More details in `jihong/keras-fcn/voc2011`


### Model Architecture

FCN8s with VGG16 as base net:

![fcn_vgg16](fcn_vgg16.png)


### TODO

 - load pre-trained weights
 - predict & test scripts
