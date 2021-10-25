import tensorflow as tf
from deepposekit.io import BaseGenerator, TrainingGenerator, DataGenerator, DLCDataGenerator
from deepposekit.models import StackedDenseNet, LEAP
import cv2

import numpy as np
from PIL import Image


from pathlib import Path

import json
import os

os.environ["CUDA_VISIBLE_DEVICES"]='2'





class OpenMonkeyDataGenerator(BaseGenerator):
    """
    OpenMonkeyDataGenerator class
    uses deepposekit.io.BaseGenerator for abstracting data loading and saving.
    Attributes that should be defined before use:
    __init__
    __len__
    compute_image_shape
    compute_keypoints_shape
    get_images
    get_keypoints
    set_keypoints (only needed for saving data)
    See docstrings for deepposekit.io.BaseGenerator for further details.
    """

    def __init__(self, annotations, image_path, size, **kwargs):
        """
        Initializes the class.
        If graph and swap_index are not defined,
        they are set to a vector of -1 corresponding
        to keypoints shape
        """
        self.samples = annotations
        self.image_path = image_path
        self.size = size
        self.height = size[0]
        self.width = size[1]

        self.n_samples = len(self.samples)
        self.num_keypoints = 17


        # TODO: Figure this graph stuff out
        # We'll use triangles as an example, so define the graph
        # -1 indicates that index is a parent node, other values >=0
        # indicate the parent for that node is the index value provided
        # Here we'll connect the two bottom points to the top
        # and avoid circular connections in the graph
        self.graph = np.array([-1, 0, 0, -1, 3, 4, 5, 6, 4, 8, 9, 4, 11, 12, 11, 14, 11])

        # You can define which keypoints to swap when the image
        # is mirrored with deepposekit.augment.FlipAxis
        # with a numpy array of index values.
        # -1 means no swap is made, but [2, 1, 0] would reverse the order
        # or [1, 0 ,2] would swap the first and second keypoints
        self.swap_index = np.ones((17)) * -1

        # This calls the BaseGenerator __init__, which does some
        # basic checks to make sure the new generator will work
        super(OpenMonkeyDataGenerator, self).__init__(**kwargs)

    def __len__(self):
        """
        Returns the number of samples in the generator as an integer
        """
        return self.n_samples

    def compute_image_shape(self):
        """
        Returns a tuple of integers describing
        the image shape in the form:
        (height, width, n_channels)
        """
        return self.get_images([1]).shape

    def compute_keypoints_shape(self):
        """
        Returns a tuple of integers describing the
        keypoints shape in the form:
        (n_keypoints, 2), where 2 is the x,y (column, row) coordinates
        """
        return self.get_keypoints([1]).shape

    def get_images(self, indexes):
        """
        Takes a list or array of indexes corresponding
        to image-keypoint pairs in the dataset.
        Returns a numpy array of images with the shape:
        (1, height, width, n_channels)
        """
        images = None
        for sample in indexes:
            sample_path = Path(self.samples[sample]["file"])
            full_path = self.image_path / sample_path
            image = np.array(Image.open(full_path))
            image = cv2.resize(image, dsize=self.size)
            image = np.expand_dims(image, axis=0)
            if images is None:
                images = image
            else:
                images = np.append(images, image, axis=0)
        assert(images.shape[0] == len(indexes))
        assert(len(images.shape) == 4)
        return images

    def get_keypoints(self, indexes):
        """
        Takes a list or array of indexes corresponding to
        image-keypoint pairs in the dataset.
        Returns a numpy array of keypoints with the shape:
        (1, n_keypoints, 2), where 2 is the x,y (column, row) coordinates
        """

        annotations = None
        for sample in indexes:
            annotation = np.array(self.samples[sample]["landmarks"]).reshape((1, self.num_keypoints, 2)).astype(np.float64)
            if annotations is None:
                annotations = annotation
            else:
                annotations = np.append(annotations, annotation, axis=0)
        return annotations

    def set_keypoints(self, indexes, keypoints):
        """
        Takes a list or array of indexes and corresponding
        to keypoints.
        Sets the values of the keypoints corresponding to the indexes
        in the dataset.
        """
        indexes = self.data_index[indexes]
        for idx in indexes:
            self.samples[idx]["landmarks"] = list(keypoints[idx].reshape(-1))



### Parameters ###


#data_generator = DLCDataGenerator(
#    project_path="/media/storage2/open_monkey"
#)

ANNOTATION_PATH = "/media/storage2/open_monkey/monkey_train_annotations.json"
IMAGE_PATH = "/media/storage2/open_monkey/train"


with open(ANNOTATION_PATH, "r") as fh:
    annotations = json.load(fh)["data"]


print(len(annotations))


data_generator = OpenMonkeyDataGenerator(annotations, IMAGE_PATH, (128, 128))

img, target = data_generator[:5]
print(img.shape.as_list())
print(target.shape.as_list())

train_generator = TrainingGenerator(data_generator, use_graph=False, downsample_factor=0)
model = LEAP(train_generator)
model.fit(batch_size=16, n_workers=8)
model.save('/path/to/saved_model.h5')
