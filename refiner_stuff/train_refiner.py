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
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from refiner_model import UNet_Lightning
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
    CHECKPOINT_PATH = "~/open_monkey_experiments/"
    EPOCHS = 10

    with open("/media/storage2/open_monkey/monkey_train_annotations.json", "r") as fh:
        annotations = json.load(fh)["data"]

    random.shuffle(annotations)

    train_annotations = annotations[:int(0.8 * len(annotations))]
    val_annotations = annotations[int(0.8 * len(annotations)):]

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
        batch_size=48,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=10,
        prefetch_factor=4
    )

    # #### Load Val Data ###

    # with open("/media/storage2/open_monkey/monkey_val_annotations.json", "r") as fh:
    #     val_annotations = json.load(fh)["data"]

    val_dataset = OpenMonkeyChallengeCropDataset(
        annotations=val_annotations,
        image_path=Path("/media/storage2/open_monkey/train"),
        bodypart_idx=BODYPART,
        crop_size=128,
        sigma=32,
        kernel_size=5
    )


    val_dataloader = DataLoader(
        val_dataset,
        batch_size=48,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=10,
        prefetch_factor=4
    )

    model = UNet_Lightning()

    ### Setup Callbacks ###
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=CHECKPOINT_PATH,
        filename=bodyparts[BODYPART] + "-{epoch:02d}-{val_loss:.2e}",
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

    trainer.fit(model, train_dataloader, val_dataloader)
