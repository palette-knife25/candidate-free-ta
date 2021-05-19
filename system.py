from options import ExperimentConfig
import pytorch_lightning as pl
import torchmetrics

import torch
from torch import nn
from torch.nn import functional as F


class CandidateFreeTE(pl.LightningModule):

  def __init__(self, opt: ExperimentConfig):
    super().__init__()

    self.opt = opt
    # Simple example
    self.net = nn.Sequential(
      nn.Flatten(),
      nn.Linear(28 * 28, 128), nn.ReLU(),
      nn.Linear(128, 256), nn.ReLU(),
      nn.Dropout(0.4),
      nn.Linear(256, 10),
      nn.LogSoftmax(dim=-1)
    )
    self.val_metrics = {"ValidationAccuracy": torchmetrics.Accuracy()}

  def forward(self, x):
      return self.net(x)

  def infere_top_k(self, x, k):
      raise Exception("Not implemented")

  def criterion(self, logits, labels):
    return F.nll_loss(logits, labels)

  def training_step(self, train_batch, batch_idx):
      x, y = train_batch
      logits = self.forward(x)
      loss = self.criterion(logits, y)

      self.log('TrainLoss', loss)
      return {'loss': loss}

  def validation_step(self, val_batch, batch_idx):
      x, y = val_batch
      logits = self.forward(x)
      loss = self.criterion(logits, y)
      return {'loss': loss, 'preds': torch.argmax(logits, dim=1), 'target': y}

  def test_step(self, val_batch, batch_idx):
      return self.validation_step(self, val_batch, batch_idx)

  def validation_step_end(self, outputs):
      for name, metric in self.val_metrics.items():
        metric(outputs['preds'], outputs['target'])
        self.log(name, metric, on_step=False, on_epoch=True)
      return outputs

  def validation_epoch_end(self, outputs):
      avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

      self.log("ValidationLoss", avg_loss)
      return {'avg_val_loss': avg_loss}

  def test_epoch_end(self, outputs):
      avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

      return {'avg_test_loss': avg_loss}

  def configure_optimizers(self):
    optimizer = [getattr(torch.optim, self.opt.optimizer)(self.parameters(), **self.opt.optimizer_args)]
    if self.opt.scheduler is not None:
      lr_scheduler = [getattr(torch.optim.lr_scheduler, self.opt.scheduler)(optimizer, **self.opt.scheduler_args)]
    else:
      lr_scheduler = []
    return optimizer, lr_scheduler
