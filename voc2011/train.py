import os
import yaml
import datetime
import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
from keras_fcn import FCN
from keras_fcn.metrics import Mean_IoU
from keras_fcn.losses import mean_categorical_crossentropy
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
                               patience=10, min_lr=0.3e-6)
early_stopper = EarlyStopping(monitor='val_loss',
                              min_delta=0.001,
                              patience=100)
csv_logger = CSVLogger(
    'output/{}_fcn_vgg16.csv'.format(datetime.datetime.now().isoformat()))


train_generator = PascalVocGenerator(image_shape=[224, 224, 3],
                                     image_resample=True,
                                     pixelwise_center=True,
                                     pixel_mean=[104.00699,
                                                 116.66877, 122.67892],
                                     pixelwise_std_normalization=True,
                                     pixel_std=[62, 62, 62])

test_generator = PascalVocGenerator(image_shape=[224, 224, 3],
                                    image_resample=True,
                                    pixelwise_center=True,
                                    pixel_mean=[104.00699,
                                                116.66877, 122.67892],
                                    pixelwise_std_normalization=True,
                                    pixel_std=[62, 62, 62])

train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])

fcn_vgg16 = FCN(input_shape=(224, 224, 3), classes=21,
                weights='imagenet', trainable_encoder=True)
optimizer = keras.optimizers.Adam(1e-3)
mean_iou = Mean_IoU(classes=21)

fcn_vgg16.compile(optimizer=optimizer,
                  loss=mean_categorical_crossentropy,
                  metrics=['accuracy'])


fcn_vgg16.fit_generator(
    train_generator.flow_from_imageset(
        class_mode='categorical',
        classes=21,
        batch_size=1,
        shuffle=True,
        image_set_loader=train_loader),
    steps_per_epoch=1112,
    epochs=100,
    validation_data=test_generator.flow_from_imageset(
        class_mode='categorical',
        classes=21,
        batch_size=1,
        shuffle=True,
        image_set_loader=val_loader),
    validation_steps=1111,
    verbose=1,
    max_q_size=100,
    callbacks=[lr_reducer, early_stopper, csv_logger, checkpointer])

fcn_vgg16.save('output/fcn_vgg16.h5')
