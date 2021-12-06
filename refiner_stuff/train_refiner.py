from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter
import json
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from refiner_pipeline import OpenMonkeyChallengeCropDataset
from refiner_model import RefinerModel, RefinerModelRegression

from fastai.vision.all import *

bodyparts = [
    "right_eye",
    "left_eye",
    "nose",
    "head",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "hip",
    "right_knee",
    "right_ankle",
    "left_knee",
    "left_ankle",
    "tail"
]


if __name__ == "__main__":

    device = torch.device("cuda:2")
    BODYPART = 2
    EPOCHS = 10

    with open("/media/storage2/open_monkey/monkey_train_annotations.json", "r") as fh:
        train_annotations = json.load(fh)["data"]

    train_dataset = OpenMonkeyChallengeCropDataset(
        annotations=train_annotations,
        image_path=Path("/media/storage2/open_monkey/train"),
        bodypart_idx=BODYPART,
        crop_size=128,
        sigma=32,
        kernel_size=5
    )


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=10,
        prefetch_factor=4
    )

    # #### Load Val Data ###

    # with open("/media/storage2/open_monkey/monkey_val_annotations.json", "r") as fh:
    #     val_annotations = json.load(fh)["data"]

    # val_dataset = OpenMonkeyChallengeCropDataset(
    #     annotations=train_annotations,
    #     image_path=Path("/media/storage2/open_monkey/val"),
    #     bodypart_idx=BODYPART,
    #     crop_size=128,
    #     sigma=32,
    #     kernel_size=5
    # )


    # val_dataloader = DataLoader(
    #     val_dataset,
    #     batch_size=16,
    #     shuffle=True,
    #     pin_memory=True,
    #     drop_last=True,
    #     num_workers=10,
    #     prefetch_factor=4
    # )

    model = Unet_Lightning()

    ### Setup Callbacks ###
    checkpoint_callback = ModelCheckpoint(
        monitor="train_f1",
        dirpath=CHECKPOINT_PATH,
        filename=bodyparts[BODYPART] + "-{epoch:02d}-{train_f1:.2f}",
        mode="max",
    )


    ### Train ###
    trainer = pl.Trainer(
        devices=torch.cuda.device_count(),
        accelerator="gpu",
        strategy="ddp",
        max_epochs=5,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_dataloader)
