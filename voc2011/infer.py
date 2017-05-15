import datetime
import numpy as np
from keras.models import load_model
from voc_generator import PascalVocGenerator, ImageSetLoader

import yaml
with open("init_args.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)


datagen = PascalVocGenerator(**init_args['pascal_voc_generator']['test'])
dataload = ImageSetLoader(**init_args['image_set_loader']['test'])

weights = 'weights.h5'
model = load_model(weights)

for fn in dataload.filenames:
    x = dataload.load_img(fn)
    x = datagen.standardize(x)
    y = model.predict(x)
    dataload.save(x, y, fn)
