import tensorflow as tf
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.contrib.data import Dataset

class ImageDataGenerator(object):
    def __init__(self, images, labels, batch_size, num_classes, shuffle=True,
                 buffer_size=1000):

        self.img_paths = images
        self.labels = labels
        self.num_classes = num_classes
        self.data_size = len(self.labels)
        self.pointer = 0

        if shuffle:
            self._shuffle_lists()
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)
        data = Dataset.from_tensor_slices((self.img_paths, self.labels))
        data = data.map(self._parse_function_train, num_threads=8,
                        output_buffer_size=100 * batch_size)

        data = data.batch(batch_size)

        self.data = data

    def _shuffle_lists(self):
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, filename, label):
        one_hot = tf.one_hot(label, self.num_classes)
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        return img_resized, one_hot

