import datetime
import numpy as np
from fcn import FCN
from voc_generator import PascalVocGenerator, ImageSetLoader
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint)

import yaml
with open("init_args.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

checkpointer = ModelCheckpoint(
    filepath="/tmp/fcn_vgg16_weights.h5",
    verbose=1,
    save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(monitor='val_loss',
                              min_delta=0.001,
                              patience=100)
csv_logger = CSVLogger(
    'output/{}_fcn_vgg16.csv'.format(datetime.datetime.now().isoformat()))


datagen = PascalVocGenerator(**init_args['pascal_voc_generator']['train'])

train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])

fcn_vgg16 = FCN(basenet='vgg16', input_shape=(500, 500, 3), num_output=21)
fcn_vgg16.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'categorical_accuracy'])

flow_args = init_args['pascal_voc_generator']['flow_from_imageset']
train_flow_args = flow_args.copy()
train_flow_args['image_set_loader'] = train_loader
val_flow_args = flow_args.copy()
val_flow_args['image_set_loader'] = val_loader

fcn_vgg16.fit_generator(datagen.flow_from_imageset(**train_flow_args),
                        steps_per_epoch=1112,
                        epochs=100,
                        validation_data=datagen.flow_from_imageset(
                            **val_flow_args),
                        validation_steps=1111,
                        verbose=1,
                        max_q_size=100,
                        callbacks=[lr_reducer, early_stopper, csv_logger])
fcn_vgg16.save('output/fcn_vgg16.h5')
