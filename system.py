from options import ExperimentConfig
import pytorch_lightning as pl
import torchmetrics

import torch
from torch import nn
from torch.nn import functional as F
from models import KBertEnricher


class CandidateFreeTE(pl.LightningModule):
	def __init__(self, opt: ExperimentConfig, tokenizer):
		super().__init__()
		self.opt = opt
		self.tokenizer = tokenizer
		self.mask_token_id = self.tokenizer.mask_token_id
		self.net = KBertEnricher(opt.base_model, opt.type_embedding_max)
		self.val_metrics = {"ValidationAccuracy": torchmetrics.Accuracy()}

	def forward(self, token_ids, type_ids, synset_ids, highway):
		return self.net(token_ids, type_ids, synset_ids, highway)

	def infere_top_k(self, x, k):
		raise Exception("Not implemented")

	def criterion(self, log_probs, labels, mask):
		"""
		NLL Loss of a batch of tokens with different lengths
		:param log_probs: torch.Tensor of shape [b x seq_len x vocab_size]
		:param labels: torch.Tensor of shape [sum(mask)]
		:param mask: torch.Tensor of shape [b x seq_len] is true if the token must be predicted
		:return:
		"""
		loss = F.nll_loss(log_probs[mask], labels, reduction="none")
		scale = (mask / mask.sum(dim=1, keepdims=True))[mask]  # weights each element in a row by the number of tokens to be predicted
		loss = (loss * scale).sum() / mask.shape[0]
		return loss

	def training_step(self, train_batch, batch_idx):
		(token_ids, type_ids, synset_ids, _, highway), gt_ids = train_batch
		log_probs = self.forward(token_ids, type_ids, synset_ids, highway)
		gt_mask = token_ids == self.mask_token_id

		loss = self.criterion(log_probs, gt_ids, gt_mask)

		self.log('TrainLoss', loss)
		return {'loss': loss}

	def validation_step(self, val_batch, batch_idx):
		(token_ids, type_ids, synset_ids, _, highway), gt_ids = val_batch
		log_probs = self.forward(token_ids, type_ids, synset_ids, highway)
		gt_mask = token_ids == self.mask_token_id

		loss = self.criterion(log_probs, gt_ids, gt_mask)
		# TODO: add validation metrics
		return {'loss': loss}

	def test_step(self, val_batch, batch_idx):
		return self.validation_step(self, val_batch, batch_idx)

	# def validation_step_end(self, outputs):
	# 	for name, metric in self.val_metrics.items():
	# 		metric(outputs['preds'], outputs['target'])
	# 		self.log(name, metric, on_step=False, on_epoch=True)
	# 	return outputs

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
			lr_scheduler = [getattr(torch.optim.lr_scheduler, self.opt.scheduler)(optimizer[0], **self.opt.scheduler_args)]
		else:
			lr_scheduler = []
		return optimizer, lr_scheduler
