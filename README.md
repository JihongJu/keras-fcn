# keras-fcn
A re-implementation of Fully Convolutional Networks with Keras

### Usage

Import FCN8s model and compile

```
from fcn import FCN8s
fcn8s = FCN8s(num_output=21, input_shape=(128, 128, 3))
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
