import numpy as np
import keras
import keras.backend as K
from keras.models import load_model
from voc_generator import PascalVocGenerator, ImageSetLoader
from keras_fcn.layers import BilinearUpSampling2D

import yaml
with open("init_args.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

datagen = PascalVocGenerator(image_shape=[224, 224, 3],
                             image_resample=True,
                             pixelwise_center=True,
                             pixel_mean=[115.85100, 110.50989, 102.16182],
                             pixelwise_std_normalization=True,
                             pixel_std=[70.30930, 69.41244, 72.60676])
dataload = ImageSetLoader(**init_args['image_set_loader']['train'])

model = load_model('/tmp/fcn_vgg16_weights.h5',
        custom_objects={'BilinearUpSampling2D': BilinearUpSampling2D})
print(model.summary())


for fn in dataload.filenames[:10]:
    x = dataload.load_img(fn)
    x = datagen.standardize(x)
    print(x.min(), x.max())
    X = x[np.newaxis, ...]
    label = dataload.load_seg(fn)
    label = np.squeeze(label, axis=-1).astype('int')
    y_enc = np.eye(21)[label]
    y_true = y_enc[np.newaxis, ...]
    result = model.evaluate(X, y_true)

    y_pred = model.predict(X)
    print(np.unique(y_true), np.unique(y_pred))
    loss = keras.losses.categorical_crossentropy(K.variable(y_true), K.variable(y_pred))
    print(y_true.shape, y_pred.shape)
    print(result, K.eval(loss))

    pred = np.argmax(y_pred, axis=-1)
    pred = pred[..., np.newaxis]
    pred = np.squeeze(pred, axis=0)
    print(np.unique(label), np.unique(pred))
    print(np.size(label), np.sum(label != 0))
