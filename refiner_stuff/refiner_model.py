import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as tm
import numpy as np

from backboned_unet import Unet


class UNet_Lightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.classifier = Unet(backbone_name="resnet18", classes=2)
        self.train_f1_score = tm.F1(num_classes=2, mdmc_average="samplewise")
        self.train_precision_score = tm.Precision(num_classes=2, mdmc_average="samplewise")
        self.train_recall_score = tm.Recall(num_classes=2, mdmc_average="samplewise")

        self.val_f1_score = tm.F1(num_classes=2, mdmc_average="samplewise")
        self.val_precision_score = tm.Precision(num_classes=2, mdmc_average="samplewise")
        self.val_recall_score = tm.Recall(num_classes=2, mdmc_average="samplewise")


    def forward(self, x):
        return self.classifier(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = F.cross_entropy(out, y.squeeze().long())
        return {'loss': loss, 'preds': out.detach(), 'targets': y}

    def training_step_end(self, outputs):
        out = outputs['preds']
        y = outputs['targets']
        self.train_f1_score(out, y.squeeze().long())
        self.train_precision_score(out, y.squeeze().long())
        self.train_recall_score(out, y.squeeze().long())
        self.log("loss", outputs["loss"])
        self.log("train_f1_score", self.train_f1_score, on_step=False, on_epoch=True)
        self.log("train_precision", self.train_precision_score, on_step=False, on_epoch=True)
        self.log("train_recall", self.train_recall_score, on_step=False, on_epoch=True)


    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = F.cross_entropy(out, y.squeeze().long())
        return {'loss': loss, 'preds': out.detach(), 'targets': y}

    def validation_step_end(self, outputs):
        out = outputs['preds']
        y = outputs['targets']
        self.val_f1_score(out, y.squeeze().long())
        self.val_precision_score(out, y.squeeze().long())
        self.val_recall_score(out, y.squeeze().long())
        self.log("val_loss", outputs["loss"])

    def validation_epoch_end(self, validation_step_outputs):
        self.log("val_f1_score", self.val_f1_score, on_step=False, on_epoch=True)
        self.log("val_precision", self.val_precision_score, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall_score, on_step=False, on_epoch=True)







