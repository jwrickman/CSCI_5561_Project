import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
from deepposekit.io import VideoReader, DataGenerator, initialize_dataset, merge_new_images
from deepposekit.annotate import KMeansSampler
import tqdm
import glob
import pandas as pd
import json
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import os



### Params ###

IMAGE_PATH = Path("/media/storage2/open_monkey/train")
### Load Images ###

with open("/media/storage2/open_monkey/monkey_train_annotations.json", "r") as fh:
        annotations = json.load(fh)["data"]

#save_points = 3000
image_size = 128
images = []
keypoints = []
skeleton = pd.read_csv("monkey_skeleton.csv")
i = 0
for annotation in tqdm(annotations):
    img = np.array(Image.open(IMAGE_PATH / annotation["file"]))
    orig_height = img.shape[0]
    orig_width = img.shape[1]
    height_scale_factor = image_size / orig_height
    width_scale_factor = image_size / orig_width
    image = cv2.resize(img, dsize=(image_size, image_size))
    keypoint = np.array(annotation["landmarks"]).reshape((17,2)).astype(np.float64)
    keypoint[:,0] *= width_scale_factor
    keypoint[:,1] *= height_scale_factor
    images.append(image)
    keypoints.append(keypoint)
    #if i == save_points - 1:
    #    images = np.array(images)
    #    keypoints = np.array(keypoints)
    #    print(images.shape)
    #    print(keypoints.shape)
    #    input()
    #    initialize_dataset(
    #        datapath = "/media/storage2/open_monkey/train_data.h5",
    #        images=images,
    #        skeleton="monkey_skeleton.csv",
    #        keypoints=keypoints,
    #        overwrite=True
    #    )
    #    del(images)
    #    del(keypoints)
    #    images = []
    #    keypoints = []
    #elif (i + 1)  % save_points == 0:
    #    merge_new_images(
    #        datapath = "/media/storage2/open_monkey/train_data.h5",
    #        merged_datapath = "/media/storage2/open_monkey/train_data_merge.h5",
    #        images=images,
    #        keypoints=keypoints,
    #        overwrite=True,
    #        mode='annotated'
    #    )
    #    os.rename("/media/storage2/open_monkey/train_data_merge.h5", "/media/storage2/open_monkey/train_data.h5")
    #    del(images)
    #    del(keypoints)
    #    images = None
    #    keypoints = None
    #i += 1
images = np.array(images)
keypoints = np.array(keypoints)
print(images.shape)
print(keypoints.shape)
initialize_dataset(
    datapath = "/media/storage2/open_monkey/train_data.h5",
    images=images,
    skeleton="monkey_skeleton.csv",
    keypoints=keypoints,
    overwrite=True
)
