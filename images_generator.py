import glob
import numpy as np


class ImagesGenerator():
    def __init__(self):
        self.train_image_path = 'train/'

        # cat = 0
        image_filenames_cat = np.array(glob.glob(self.train_image_path + 'cat.*.jpg'))
        image_filenames_dog = np.array(glob.glob(self.train_image_path + 'dog.*.jpg'))

        self.image_path = []
        self.label_path = []

        for catitem in image_filenames_cat:
            self.image_path.append(catitem)
            self.label_path.append(0)
        for dogitem in image_filenames_dog:
            self.image_path.append(dogitem)
            self.label_path.append(1)

    def toList(self):
        return self.image_path, self.label_path
