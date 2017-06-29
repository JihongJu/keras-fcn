import os
import yaml
import datetime
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras_fcn import FCN
from voc_generator import PascalVocGenerator, ImageSetLoader
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

global _SESSION
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
_SESSION = tf.Session(config=config)
K.set_session(_SESSION)


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


datagen = PascalVocGenerator(image_shape=[500, 500, 3],
                             image_resample=True,
                             pixelwise_center=True,
                             pixel_mean=[104.00699, 116.66877, 122.67892])


train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])

fcn_vgg16 = FCN(input_shape=(500, 500, 3), classes=21,
                weights='imagenet', trainable_encoder=True)
fcn_vgg16.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'categorical_accuracy'])


fcn_vgg16.fit_generator(
    datagen.flow_from_imageset(class_mode='categorical',
                               classes=21,
                               batch_size=1,
                               shuffle=True,
                               image_set_loader=train_loader),
    steps_per_epoch=1112,
    epochs=100,
    validation_data=datagen.flow_from_imageset(class_mode='categorical',
                                               classes=21,
                                               batch_size=1,
                                               shuffle=True,
                                               image_set_loader=val_loader),
    validation_steps=1111,
    verbose=1,
    max_q_size=100,
    callbacks=[lr_reducer, early_stopper, csv_logger])
fcn_vgg16.save('output/fcn_vgg16.h5')
