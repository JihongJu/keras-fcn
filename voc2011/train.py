import os
import yaml
import datetime
import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
from keras_fcn import FCN
from voc_generator import PascalVocGenerator, ImageSetLoader
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN)
from keras_fcn.callbacks import CheckNumericsOps


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
                               patience=10, min_lr=1e-12)
early_stopper = EarlyStopping(monitor='val_loss',
                              min_delta=0.001,
                              patience=30)
nan_terminator = TerminateOnNaN()
csv_logger = CSVLogger(
    #'output/{}_fcn_vgg16.csv'.format(datetime.datetime.now().isoformat()))
    'output/tmp_fcn_vgg16.csv')
#check_num = CheckNumericsOps(validation_data=[np.random.random((1, 224, 224, 3)), 1],
#                             histogram_freq=100)


datagen = PascalVocGenerator(image_shape=[224, 224, 3],
                                    image_resample=True,
                                    pixelwise_center=True,
                                    pixel_mean=[115.85100, 110.50989, 102.16182],
                                    pixelwise_std_normalization=True,
                                    pixel_std=[70.30930, 69.41244, 72.60676])

train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])

fcn_vgg16 = FCN(input_shape=(224, 224, 3), classes=21, weight_decay=3e-3,
                weights='imagenet', trainable_encoder=True)
optimizer = keras.optimizers.Adam(1e-4)

fcn_vgg16.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

fcn_vgg16.fit_generator(
    datagen.flow_from_imageset(
        class_mode='categorical',
        classes=21,
        batch_size=1,
        shuffle=True,
        image_set_loader=train_loader),
    steps_per_epoch=1112,
    epochs=100,
    validation_data=datagen.flow_from_imageset(
        class_mode='categorical',
        classes=21,
        batch_size=1,
        shuffle=True,
        image_set_loader=val_loader),
    validation_steps=1111,
    verbose=1,
    callbacks=[lr_reducer, early_stopper, csv_logger, checkpointer, nan_terminator])

fcn_vgg16.save('output/fcn_vgg16.h5')
