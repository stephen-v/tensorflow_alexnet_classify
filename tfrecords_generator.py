"""
this script is used to generate tfrecords. it contains a function,
the input data is a map.
example:  key: path/to/your/images  value: 0
"""

import tensorflow as tf


class TFRecordsGenerator(object):
    def __init__(self, images, out_path):
        """
        :param images:
        :param out_path
        """
        self.images = images
        self.out_path = out_path

    def generate(self, size):
        """
        start to generate tfrecords from url_map
        :param size:image size [227,227]
        :return:
        """
        writer = None
        current_i = 0
        with tf.Session() as sess:
            for image in self.images:
                current_i += 1
                image_url = image['image_path']
                record_filename = "{path}/{current_index}.tfrecords".format(path=self.out_path,
                                                                            current_index=current_i)
                try:
                    writer = tf.python_io.TFRecordWriter(record_filename)
                    image_stream = tf.read_file(image_url)
                    image_raw = tf.image.decode_jpeg(image_stream, channels=3)
                    image_raw = tf.image.resize_images(image_raw, size=size)
                    resized_image = sess.run(tf.cast(image_raw, tf.uint8)).tobytes()
                    image_lable = sess.run(tf.cast(image['type'],tf.uint8)).tobytes()

                except:
                    print(image_url)
                    continue
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_lable])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[resized_image]))
                }))
                writer.write(example.SerializeToString())

    def read_records(self, batch_size):
        """
        read records from out_path
        :param batch_size:
        :return:image_batch, label_batch
        """
