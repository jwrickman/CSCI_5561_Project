from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter
import json
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class OpenMonkeyChallengeCropDataset(Dataset):

    def __init__(self,
                 annotations,
                 image_path,
                 bodypart_idx,
                 crop_size,
                 sigma):
        self.annotations = annotations
        self.image_path = Path(image_path)
        self.bodypart_idx = bodypart_idx
        self.crop_size = crop_size
        self.half_crop_size = crop_size // 2
        self.sigma = sigma


    def __len__(self):
        return len(self.annotations)


    def create_confidence_map(self, offset_xy):
        mask = np.zeros((self.crop_size, self.crop_size), dtype=np.float64)
        mask[self.half_crop_size + -1 * offset_xy[1],self.half_crop_size + -1 * offset_xy[0]] = 1
        mask = gaussian_filter(mask, self.sigma)
        #mask = 1 / np.max(mask)
        return mask


    def crop_image(self, image, crop_center):
        left = crop_center[0] - self.half_crop_size
        top = crop_center[1] - self.half_crop_size
        right = crop_center[0] + self.half_crop_size
        bottom = crop_center[1] + self.half_crop_size
        return image.crop((left, top, right, bottom))

    def show_image(self, idx):
        sample = self.annotations[idx]

        image = Image.open(self.image_path / sample["file"])
        keypoints = np.array(sample["landmarks"]).reshape((17,2))
        scale_w = 128 / np.array(image).shape[0]
        scale_h = 128 / np.array(image).shape[1]

        plt.imshow(image.resize((128,128)))
        #plt.scatter((keypoints[self.bodypart_idx,0] + 20) * scale_h, (keypoints[self.bodypart_idx,1] + 15) * scale_w, c="red", s=20, cmap=plt.cm.hsv, zorder=3)
        #plt.scatter(keypoints[self.bodypart_idx,0] * scale_h, keypoints[self.bodypart_idx,1] * scale_w, c="blue", s=20, marker='+', cmap=plt.cm.hsv, zorder=3)
        plt.savefig("visualizations/clean_compressed_error{}.png".format(idx))
        plt.clf()
        plt.cla()


        plt.imshow(image)
        plt.scatter(keypoints[self.bodypart_idx,0] + 20, keypoints[self.bodypart_idx,1] + 15, c="red", s=20, cmap=plt.cm.hsv, zorder=3)
        plt.scatter(keypoints[self.bodypart_idx,0], keypoints[self.bodypart_idx,1], c="blue", s=20, marker='+', cmap=plt.cm.hsv, zorder=3)
        plt.savefig("visualizations/clean_full_error{}.png".format(idx))
        plt.clf()
        plt.cla()

    def get_crop_and_show(self, idx):
        sample = self.annotations[idx]

        image = Image.open(self.image_path / sample["file"])

        keypoint_xy = np.array(sample["landmarks"]).reshape((17, 2))[self.bodypart_idx]

        offset_xy = np.random.randint(-self.half_crop_size + 1, self.half_crop_size, (2))
        offset_xy = np.array([15,20])

        image_crop = np.array(self.crop_image(image, keypoint_xy + offset_xy))

        plt.imshow(image_crop)
    #    plt.scatter(x=self.half_crop_size + -1 * offset_xy[0], y=self.half_crop_size + -1 * offset_xy[1], c="blue", s=10, marker="+")
    #    plt.scatter(x=self.half_crop_size, y=self.half_crop_size, c="red", s=10)
        keypoints = np.array([self.half_crop_size + -1 * offset_xy[1],self.half_crop_size + -1 * offset_xy[0]])
        plt.savefig("visualizations/clean_random_crop_{}.png".format(idx))
        plt.clf()
        plt.cla()




        # Normalize image (imagenet)

        image_crop = image_crop / 255
        image_crop[:,:,0] = (image_crop[:,:,0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        image_crop[:,:,1] = (image_crop[:,:,1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
        image_crop[:,:,2] = (image_crop[:,:,2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]


        target_crop = self.create_confidence_map(offset_xy)
        image_crop = torch.from_numpy(np.array(image_crop, dtype=np.float32)).permute(2, 0, 1)

        plt.imshow(target_crop)
    #    plt.scatter(x=self.half_crop_size + -1 * offset_xy[0], y=self.half_crop_size + -1 * offset_xy[1], c="blue", marker="+")
    #    plt.scatter(x=self.half_crop_size, y=self.half_crop_size, c="red", marker="o", s=5)
        plt.savefig("visualizations/clean_crop_confidence_map_crosses_{}.png".format(idx))
        plt.clf()
        plt.cla()

        target_crop = torch.from_numpy(target_crop).float()

        return image_crop, target_crop


    def __getitem__(self, idx):
        sample = self.annotations[idx]

        image = Image.open(self.image_path / sample["file"])

        keypoint_xy = np.array(sample["landmarks"]).reshape((17, 2))[self.bodypart_idx]

        offset_xy = np.random.randint(-self.half_crop_size + 1, self.half_crop_size, (2))

        image_crop = np.array(self.crop_image(image, keypoint_xy + offset_xy))

        # Normalize image (imagenet)

        image_crop = image_crop / 255
        image_crop[:,:,0] = (image_crop[:,:,0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        image_crop[:,:,1] = (image_crop[:,:,1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
        image_crop[:,:,2] = (image_crop[:,:,2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]


        #plt.imshow(image_crop)
        #plt.scatter(x=self.half_crop_size + -1 * offset_xy[0], y=self.half_crop_size + -1 * offset_xy[1])
        #plt.show()

        target_crop = self.create_confidence_map(offset_xy)
        image_crop = torch.from_numpy(np.array(image_crop, dtype=np.float32)).permute(2, 0, 1)

        target_crop = torch.from_numpy(target_crop).float()

#        keypoints = np.array([self.half_crop_size + -1 * offset_xy[1],self.half_crop_size + -1 * offset_xy[0]])
#        keypoints = torch.from_numpy(keypoints).float()

        return image_crop, target_crop


if __name__ == "__main__":
    with open("/media/storage2/open_monkey/monkey_train_annotations.json", "r") as fh:
        annotations = json.load(fh)["data"]


    dataset = OpenMonkeyChallengeCropDataset(
        annotations=annotations,
        image_path=Path("/media/storage2/open_monkey/train"),
        bodypart_idx=2,
        crop_size=150,
        sigma=32,
    )
    dataset.show_image(0)
    dataset.get_crop_and_show(0)

    """
    for i in range(10):
        dataset.show_image(i)
        dataset.get_crop_and_show(i)
    """
