# keras-fcn
A re-implementation of Fully Convolutional Networks with Keras

### Installation

This implementation is based on Keras together with the GPU version of Tensorflow. Therefore, it is highly recommended to use a container, e.g. [Docker](https://www.docker.com/), to run the training processes.

The following installation procedures assumes Nvidia driver, [docker](https://docs.docker.com/engine/installation/linux/ubuntu/) and [nvidia-docker](https://devblogs.nvidia.com/parallelforall/nvidia-docker-gpu-server-application-deployment-made-easy/) are properly installed on a Ubuntu machine.

First clone the repository:

```
git clone https://github.com/JihongJu/keras-fcn.git
```

Start bash in a docker image and mount the local repository to `/workspace`
```
$ nvidia-docker run -it --rm -v `pwd`/keras-fcn/:/workspace jihong/nvidia-keras bash
```

Validate installation by by starting a train process

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

Prepare the train set

```
import numpy as np
X_train = np.random.rand(10, 128, 128, 3)
labels = np.random.randint(0, 21, size=[10, 128, 128])
y_train = np.eye(21)[labels]
```


Train

```
fcn8s.fit(X_train, y_train, batch_size=1)
```
