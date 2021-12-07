import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as tm
import numpy as np

from backboned_unet import Unet


class UNet_Lightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.classifier = Unet(backbone_name="resnet18", classes=1)

        self.val_f1_score = tm.F1(num_classes=2, mdmc_average="samplewise", average=None)
        self.val_precision_score = tm.Precision(num_classes=2, mdmc_average="samplewise", average=None)
        self.val_recall_score = tm.Recall(num_classes=2, mdmc_average="samplewise", average=None)


    def forward(self, x):
        out = self.classifier(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = F.l1_loss(out.squeeze(), y.squeeze())
        return {'loss': loss, 'preds': out.detach(), 'targets': y}

    def training_step_end(self, outputs):
        out = outputs['preds']
        y = outputs['targets']
        self.log("loss", outputs["loss"])


    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = F.l1_loss(out.squeeze(), y.squeeze())
        return {'loss': loss, 'preds': out.detach(), 'targets': y}

    def validation_step_end(self, outputs):
        out = outputs['preds']
        y = outputs['targets']
        self.log("val_loss", outputs["loss"])
#        y[y > 0.0002] = 1
        #self.val_f1_score(out, y.squeeze().long())
        #self.val_precision_score(out, y.squeeze().long())
        #self.val_recall_score(out, y.squeeze().long())

#    def validation_epoch_end(self, validation_step_outputs):
        #f1_score = self.val_f1_score.compute()
        #precision_score = self.val_precision_score.compute()
        #recall_score = self.val_recall_score.compute()

        #self.log("val_f1_score", f1_score)
        #self.log("val_precision", precision_score)
        #self.log("val_recall", recall_score)

#        self.val_f1_score.reset()
#        self.val_precision_score.reset()
#        self.val_recall_score.reset()
