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


if __name__ == "__main__":

    device = torch.device("cuda:2")
    EPOCHS = 10

    with open("/media/storage2/open_monkey/monkey_train_annotations.json", "r") as fh:
        annotations = json.load(fh)["data"]


    dataset = OpenMonkeyChallengeCropDataset(
        annotations=annotations,
        image_path=Path("/media/storage2/open_monkey/train"),
        bodypart_idx=2,
        crop_size=128,
        sigma=32,
    )

    #dataset.show_image(0)

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=10,
        prefetch_factor=4
    )

    model = RefinerModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))

    model = create_unet_model(arch=models.resnet18, img_size=(128,128), n_out=1, pretrained=True)
    model.to(device)


    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0

        for x, y in tqdm(dataloader):
            out = model(x.to(device))
            optimizer.zero_grad()
            loss = loss_fn(out.squeeze(), y.to(device))

            loss.backward()
            optimizer.step()
            epoch_loss += loss.data
        print("Epoch: {}, train_loss: {}".format(epoch, epoch_loss/len(dataloader)))


    model.eval()
    for i in range(10):
        img, target = dataset.get_crop_and_show(1)
        out = model(torch.unsqueeze(img, 0).to(device))
        out = out.squeeze().cpu().detach().numpy()
        plt.imshow(out)
        #plt.scatter(x=out[0], y=out[1], c="blue")
        #plt.scatter(x=out[1], y=out[0], c="green")
        plt.show()






