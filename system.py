from options import ExperimentConfig
import pytorch_lightning as pl

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

  def forward(self, x):
      return self.net(x)

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
      acc = (torch.argmax(logits, dim=-1) == y).float().mean()
      return {'loss': loss, 'accuracy': acc}

  def test_step(self, val_batch, batch_idx):
      return self.validation_step(self, val_batch, batch_idx)

  def validation_epoch_end(self, outputs):
      avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
      avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()

      self.log("AvgValLoss", avg_loss)
      self.log("AvgValAccuracy", avg_acc)
      return {'avg_val_loss': avg_loss, 'avg_val_accuracy': avg_acc}

  def test_epoch_end(self, outputs):
      avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
      avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()

      return {'avg_test_loss': avg_loss, 'avg_test_accuracy': avg_acc}

  def configure_optimizers(self):
    optimizer = getattr(torch.optim, self.opt.optimizer)(self.parameters(), **self.opt.optimizer_args)
    lr_scheduler = getattr(torch.optim.lr_scheduler, self.opt.scheduler)(optimizer, **self.opt.scheduler_args)
    return [optimizer], [lr_scheduler]
