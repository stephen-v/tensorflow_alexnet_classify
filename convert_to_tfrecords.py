import glob
import numpy as np
from tfrecords_generator import TFRecordsGenerator

train_image_path = 'train/'
image_filenames_cat = np.array(glob.glob(train_image_path + 'cat.*.jpg'))
image_filenames_dog = np.array(glob.glob(train_image_path + 'dog.*.jpg'))

images = []


def add_images(image_sets, classify, images):
    for i in image_sets:
        image_path = train_image_path + i.split('\\')[1]
        image = {'image_path': image_path, 'type': classify}
        images.append(image)


add_images(image_filenames_cat, 0, images)
add_images(image_filenames_dog, 0, images)

tf_generator = TFRecordsGenerator(images, './tfrecords')
tf_generator.generate([227, 227])
