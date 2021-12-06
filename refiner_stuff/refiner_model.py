import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as tm
import numpy as np

from backboned_unet import Unet


class UNet_Lightning(pl.LightningModule):
    def __init__(self, metrics):
        super().__init__()
        self.classifier = Unet(backbone_name="resnet18", classes=2)
        self.f1_score = tm.F1(num_classes=2)

    def forward(self, x):
        return self.classifier(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = F.cross_entropy(out, y.squeeze().long())
        f1_score = self.f1_score(out, y.squeeze().long())
        self.log("train_loss", loss)
        self.log("train_f1", f1_score)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = F.cross_entropy(out, y.squeeze().long())
        F1 = self.f1_score(out, y.squeeze().long())
        self.log("val_loss", loss)
        self.log("val_f1", F1)
        return loss
