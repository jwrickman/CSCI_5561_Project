from deepposekit.io import BaseGenerator, TrainingGenerator
from deepposekit.io import DataGenerator, TrainingGenerator
from deepposekit.models import StackedDenseNet

from Pathlib.Path import Path

import json


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

    def __init__(self, annotations, image_path **kwargs):
        """
        Initializes the class.
        If graph and swap_index are not defined,
        they are set to a vector of -1 corresponding
        to keypoints shape
        """
        self.samples = annotations
        self.image_path = image_path

        self.n_samples = len(self.samples)
        self.num_keypoints = 17


        # TODO: Figure this graph stuff out
        # We'll use triangles as an example, so define the graph
        # -1 indicates that index is a parent node, other values >=0
        # indicate the parent for that node is the index value provided
        # Here we'll connect the two bottom points to the top
        # and avoid circular connections in the graph
        self.graph = np.array([-1, 0, 0])

        # You can define which keypoints to swap when the image
        # is mirrored with deepposekit.augment.FlipAxis
        # with a numpy array of index values.
        # -1 means no swap is made, but [2, 1, 0] would reverse the order
        # or [1, 0 ,2] would swap the first and second keypoints
        self.swap_index = np.array([-1, -1, -1])

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
        return self.get_images(1).shape

    def compute_keypoints_shape(self):
        """
        Returns a tuple of integers describing the
        keypoints shape in the form:
        (n_keypoints, 2), where 2 is the x,y (column, row) coordinates
        """
        return self.get_keypoints(1).shape

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
            image = Image.open(full_path)
            if images is None:
                images = np.expand_dims(image, axis=0)
            else:
                np.append(images, image, axis=0)
        assert(images.shape[0] == len(indexes))
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
            annotation = np.array(self.samples[sample]["landmarks"]).reshape(self.num_keypoints, 2)
            if annotations is None:
                annotations = np.expand_dims(annotation, axis=0)
            else:
                np.append(annotations, annotation, axis=0)
        return self.keypoints[indexes]

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

ANNOTATION_PATH = train_annotations.json
IMAGE_PATH = "train_images"


with open(ANNOTATION_PATH, "r") as fh:
    annotations = json.load(fh)

print(len(annotations))


data_generator = OpenMonkeyDataGenerator(annotations, image_path)
train_generator = TrainingGenerator(data_generator)
model = StackedDenseNet(train_generator)
model.fit(batch_size=16, n_workers=8)
model.save('/path/to/saved_model.h5')



