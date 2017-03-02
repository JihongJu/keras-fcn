import os
import numpy as np
from keras import backend as K
from keras.preprocessing.image import (
    ImageDataGenerator,
    Iterator,
    img_to_array,
    array_to_img)
try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None


class ImageSegmentationGenerator(ImageDataGenerator):
    """A real-time data augmentation generator for PASCAL VOC2011 Segmentation
    """
    def flow_from_imageset(self, image_set, image_data_loader,
                           segmentation_data_loader,
                           batch_size=1, shuffle=True, seed=None):
        return ImageSetIterator(
            image_set, self,
            image_data_loader=image_data_loader,
            segmentation_data_loader=segmentation_data_loader,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed)


class ImageDataLoader(object):
    """Helper class to load image data into numpy arrays
    """
    def __init__(self, image_dir, image_format='jpg', color_mode='rgb',
                 target_size=(224, 224), dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpg'):
        self.image_dir = image_dir
        self.image_dir = image_dir
        if not os.path.exists(image_dir):
            raise IOError('Directory {} does not exist. Please provide a valid'
                          'directory.'.format(image_dir))

        self.image_format = image_format
        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
        if self.image_format not in white_list_formats:
            raise ValueError('Invalid image format:', image_format,
                             '; expected "png", "jpg", "jpeg" or "bmp"')
        self.color_mode = color_mode
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        if dim_ordering not in {'default', 'tf', 'th'}:
            raise ValueError('Invalid dim ordering:', dim_ordering,
                             '; expected "default", "tf" or "th".')
        self.dim_ordering = dim_ordering
        if dim_ordering == 'default':
            self.dim_ordering = K.image_dim_ordering()
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.grayscale = self.color_mode == 'grayscale'

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

    def load(self, fn):
        """Image load method.
        # Arguments
            fn: filename of the image (without extension suffix)
        # Returns
            arr: numpy array of shape self.image_shape
        """
        img_path = os.path.join(self.image_dir,
                                '{}.{}'.format(fn, self.image_format))
        if not os.path.exists(img_path):
            raise IOError('Image {} does not exist.'.format(img_path))
        img = load_img(img_path, self.grayscale, self.target_size)
        arr = img_to_array(img, dim_ordering=self.dim_ordering)

        return arr

    def save(self, arr, index):
        """Image save method
        """
        if self.save_to_dir:
            img = array_to_img(arr, self.dim_ordering, scale=True)
            fname = '{prefix}_{index}_{hash}.{format}'.format(
                prefix=self.save_prefix,
                index=index,
                hash=np.random.randint(1e4),
                format=self.save_format)
            img.save(os.path.join(self.save_to_dir, fname))


class ImageSetIterator(Iterator):

    def __init__(self, image_set, image_data_generator,
                 image_data_loader, segmentation_data_loader,
                 batch_size=1, shuffle=False, seed=None):
        self.image_set = image_set
        if not os.path.exists(image_set):
            raise IOError('File {} does not exist. Please provide a file '
                          'that contains an image filename '
                          '(without extension) per line.'.format(image_set))

        self.image_data_generator = image_data_generator
        self.image_data_loader = image_data_loader
        self.segmentation_data_loader = segmentation_data_loader

        self.filenames = np.loadtxt(image_set, dtype=str)
        self.nb_sample = len(self.filenames)
        for fn in self.filenames:
            x = self.image_data_loader.load(fn)
            y = self.segmentation_data_loader.load(fn).astype(np.uint8)
            if not x.shape[:2] == y.shape[:2] \
                    and not x.shape[-2:] == y.shape[-2:]:
                raise IOError('Segmentation does not match the shape '
                              'of Image for {}.'.format(fn))

        super(ImageSetIterator, self).__init__(self.nb_sample, batch_size,
                                               shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)
        batch_x = np.zeros(
            (current_batch_size,) + self.image_data_loader.image_shape,
            dtype=K.floatx())
        batch_y = np.zeros(
            (current_batch_size,)
            + self.segmentation_data_loader.image_shape,
            dtype=np.int8)

        for i, j in enumerate(index_array):
            fn = self.filenames[j]
            print(fn)
            x = self.image_data_loader.load(fn)
            batch_x[i] = x
            y = self.segmentation_data_loader.load(fn).astype(np.uint8)
            batch_y[i] = y

        # save augmented images to disk for debugging
        if self.image_data_loader.save_to_dir:
            for i in range(current_batch_size):
                self.image_data_loader.save(batch_x[i], current_index+i)
        if self.segmentation_data_loader.save_to_dir:
            for i in range(current_batch_size):
                self.image_data_loader.save(batch_y[i], current_index+i)

        return batch_x, batch_y


def load_img(path, grayscale=False, target_size=None):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:  # Keep the original value for Segmentation
        pass
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img
