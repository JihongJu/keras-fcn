import numpy as np
from fcn import FCN

fcn8s = FCN(basenet='vgg16', input_shape=(224, 224, 3), num_output=21)
fcn8s.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

X_train = np.random.rand(10, 224, 224, 3)
labels = np.random.randint(0, 21, size=[10, 224, 224])
y_train = np.eye(21)[labels]

fcn8s.fit(X_train, y_train, batch_size=1)
