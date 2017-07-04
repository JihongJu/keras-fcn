"""Pascal VOC Segmenttion Generator."""
from __future__ import unicode_literals
import os
import numpy as np
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import (
    ImageDataGenerator,
    Iterator,
    load_img,
    img_to_array,
    pil_image,
    array_to_img)


class PascalVocGenerator(ImageDataGenerator):
    """A real-time data augmentation generator for PASCAL VOC2011."""

    def __init__(self,
                 image_shape=(500, 500, 3),
                 image_resample=True,
                 pixelwise_center=False,
                 pixel_mean=(0., 0., 0.),
                 pixelwise_std_normalization=False,
                 pixel_std=(1., 1., 1.),
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None):
        """Init."""
        self.image_shape = tuple(image_shape)
        self.image_resample = image_resample
        self.pixelwise_center = pixelwise_center
        self.pixel_mean = np.array(pixel_mean)
        self.pixelwise_std_normalization = pixelwise_std_normalization
        self.pixel_std = np.array(pixel_std)
        super(PascalVocGenerator, self).__init__()

    def standardize(self, x):
        """Standardize image."""
        if self.pixelwise_center:
            x -= self.pixel_mean
        if self.pixelwise_std_normalization:
            x /= self.pixel_std
        return super(PascalVocGenerator, self).standardize(x)

    def flow_from_imageset(self, image_set_loader,
                           class_mode='categorical', classes=None,
                           batch_size=1, shuffle=True, seed=None):
        """PascalVocGenerator."""
        return IndexIterator(
            image_set_loader, self,
            class_mode=class_mode,
            classes=classes,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed)


class IndexIterator(Iterator):
    """Iterator over index."""

    def __init__(self, image_set_loader, image_data_generator,
                 class_mode='categorical', classes=None,
                 batch_size=1, shuffle=False, seed=None):
        """Init."""
        self.image_set_loader = image_set_loader
        self.image_data_generator = image_data_generator

        self.filenames = image_set_loader.filenames
        self.image_shape = image_set_loader.image_shape

        self.classes = classes
        if class_mode == 'binary':
            label_shape = list(self.image_shape).pop(self.channel_axis - 1)
            self.label_shape = tuple(label_shape)
        elif class_mode == 'categorical':
            label_shape = list(self.image_shape)
            label_shape[self.image_data_generator.channel_axis - 1] \
                = self.classes
            self.label_shape = tuple(label_shape)

        super(IndexIterator, self).__init__(len(self.filenames), batch_size,
                                            shuffle, seed)

    def next(self):
        """Next batch."""
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)
        batch_x = np.zeros(
            (current_batch_size,) + self.image_shape,
            dtype=K.floatx())
        batch_y = np.zeros(
            (current_batch_size,) + self.label_shape,
            dtype=np.int8)
        #batch_y = np.reshape(batch_y, (current_batch_size, -1, self.classes))

        for i, j in enumerate(index_array):
            fn = self.filenames[j]
            x = self.image_set_loader.load_img(fn)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            y = self.image_set_loader.load_seg(fn)
            y = to_categorical(y, self.classes).reshape(self.label_shape)
            #y = np.reshape(y, (-1, self.classes))
            batch_y[i] = y

        # save augmented images to disk for debugging
        #if self.image_set_loader.save_to_dir:
        #    for i in range(current_batch_size):
        #        x = batch_x[i]
        #        y = batch_y[i].argmax(
        #            self.image_data_generator.channel_axis - 1)
        #        if self.image_data_generator.data_format == 'channels_first':
        #            y = y[np.newaxis, ...]
        #        else:
        #            y = y[..., np.newaxis]
        #        self.image_set_loader.save(x, y, current_index + i)

        return batch_x, batch_y


class ImageSetLoader(object):
    """Helper class to load image data into numpy arrays."""

    def __init__(self, image_set, image_dir, label_dir, target_size=(500, 500),
                 image_format='jpg', color_mode='rgb', label_format='png',
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='jpg'):
        """Init."""
        if data_format is None:
            data_format = K.image_data_format()
        self.data_format = data_format
        self.target_size = tuple(target_size)

        if not os.path.exists(image_set):
            raise IOError('Image set {} does not exist. Please provide a'
                          'valid file.'.format(image_set))
        self.filenames = np.loadtxt(image_set, dtype=bytes)
        try:
            self.filenames = [fn.decode('utf-8') for fn in self.filenames]
        except AttributeError as e:
            print(str(e), self.filenames[:5])
        if not os.path.exists(image_dir):
            raise IOError('Directory {} does not exist. Please provide a '
                          'valid directory.'.format(image_dir))
        self.image_dir = image_dir
        if label_dir and not os.path.exists(label_dir):
            raise IOError('Directory {} does not exist. Please provide a '
                          'valid directory.'.format(label_dir))
        self.label_dir = label_dir

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
        self.image_format = image_format
        if self.image_format not in white_list_formats:
            raise ValueError('Invalid image format:', image_format,
                             '; expected "png", "jpg", "jpeg" or "bmp"')
        self.label_format = label_format
        if self.label_format not in white_list_formats:
            raise ValueError('Invalid image format:', label_format,
                             '; expected "png", "jpg", "jpeg" or "bmp"')

        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        self.grayscale = self.color_mode == 'grayscale'

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

    def load_img(self, fn):
        """Image load method.

        # Arguments
            fn: filename of the image (without extension suffix)
        # Returns
            arr: numpy array of shape self.image_shape
        """
        img_path = os.path.join(self.image_dir,
                                '{}.{}'.format(fn,
                                               self.image_format))
        if not os.path.exists(img_path):
            raise IOError('Image {} does not exist.'.format(img_path))
        img = load_img(img_path, self.grayscale, self.target_size)
        x = img_to_array(img, data_format=self.data_format)

        return x

    def load_seg(self, fn):
        """Segmentation load method.

        # Arguments
            fn: filename of the image (without extension suffix)
        # Returns
            arr: numpy array of shape self.target_size
        """
        label_path = os.path.join(self.label_dir,
                                  '{}.{}'.format(fn, self.label_format))
        img = pil_image.open(label_path)
        if self.target_size:
            wh_tuple = (self.target_size[1], self.target_size[0])
        if img.size != wh_tuple:
            img = img.resize(wh_tuple)
        y = img_to_array(img, self.data_format)
        y[y == 255] = 0

        return y

    def save(self, x, y, index):
        """Image save method."""
        img = array_to_img(x, self.data_format, scale=True)
        mask = array_to_img(y, self.data_format, scale=True)
        img.paste(mask, (0, 0), mask)

        fname = 'img_{prefix}_{index}_{hash}.{format}'.format(
            prefix=self.save_prefix,
            index=index,
            hash=np.random.randint(1e4),
            format=self.save_format)
        img.save(os.path.join(self.save_to_dir, fname))
