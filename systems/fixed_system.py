import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from metrics import PrecisionK, MeanAveragePrecision
from metrics.mrr import build_match_matrix, MeanReciprocalRank
from models.fixed_enricher import FixedEnricher, FixedTokenizer
from options import ExperimentConfig


class FixedTESystem(pl.LightningModule):
	def __init__(self, opt: ExperimentConfig, token_embedder: FixedTokenizer):
		super().__init__()
		self.opt = opt
		self.mask_token_id = token_embedder.mask_token_id
		self.net = FixedEnricher(token_embedder, type_embedding_max=opt.type_embedding_max)
		self.val_metrics = {"ValidationMRR": MeanReciprocalRank(),
							"ValidationPrecision@10": PrecisionK(),
							"ValidationMAP": MeanAveragePrecision()}

	def forward(self, token_ids, type_ids, synset_ids, highway):
		return self.net(token_ids, type_ids, synset_ids, highway)

	def insert_mask_tokens(self, token_ids, type_ids, synset_ids, highway, n_tokens, position):
		batch_size = token_ids.shape[0]
		token_ids = torch.cat([token_ids[:, :position],
							   token_ids.new_ones(batch_size, n_tokens) * self.mask_token_id,
							   token_ids[:, position:, ]], dim=1)
		type_ids = torch.cat([type_ids[:, :position],
							  type_ids.new_zeros(batch_size, n_tokens),
							  type_ids[:, position:, ]], dim=1)
		synset_ids = torch.cat([synset_ids[:, :position],
								synset_ids.new_zeros(batch_size, n_tokens),
								synset_ids[:, position:, ]], dim=1)
		highway = torch.cat([highway[:, :position],
							 highway.new_ones(batch_size, n_tokens),
							 highway[:, position:, ]], dim=1)
		return token_ids, type_ids, synset_ids, highway

	@torch.no_grad()
	def infere_top_k(self, batch, k):
		if len(batch) == 2:
			batch = batch[0]
		token_ids, type_ids, synset_ids, _, highway = batch
		token_ids, type_ids, synset_ids, highway = self.insert_mask_tokens(token_ids, type_ids, synset_ids, highway,
																		   n_tokens=1, position=0)

		pred_emb = self.forward(token_ids, type_ids, synset_ids, highway)[:, 0, :]
		pred_emb = pred_emb / torch.linalg.norm(pred_emb, dim=1, keepdim=True)  # b  x hdim
		offset = self.net.token_embedder.reserved
		vectors = self.net.token_embedder.embedding.weight[offset:, :]  # v x hdim
		vectors = vectors / torch.linalg.norm(vectors, dim=1, keepdim=True)  # normalize for cos similarity
		scores = pred_emb @ vectors.T  # b x v
		topk = torch.topk(scores, k, dim=1, largest=True, sorted=True).indices + offset  # b x k
		return topk

	def criterion(self, pred_emb, labels, mask):
		"""
		NLL Loss of a batch of tokens with different lengths
		:param pred_emb: torch.Tensor of shape [b x hdim]
		:param labels: torch.Tensor of shape [b x hdim]
		:param mask: torch.Tensor of shape [b x seq_len] is true if the token must be predicted
		:return:
		"""
		loss = F.mse_loss(pred_emb, labels, reduction="none").mean(dim=1)
		loss = loss[mask].sum() / max(mask.sum().item(), 1)
		return loss

	def training_step(self, train_batch, batch_idx):
		(token_ids, type_ids, synset_ids, _, highway), gt_ids = train_batch
		pred_emb = self.forward(token_ids, type_ids, synset_ids, highway)[:, 0, :]
		gt_emb = self.net.embedder.token_embedding(gt_ids)
		ukw_mask = gt_ids == self.net.token_embedder.ukw_idx

		loss = self.criterion(pred_emb, gt_emb, ~ukw_mask)

		self.log('TrainLoss', loss)
		return {'loss': loss}

	def validation_step(self, val_batch, batch_idx):
		(token_ids, type_ids, synset_ids, _, highway), gt_ids = val_batch
		# pred_emb = self.forward(token_ids, type_ids, synset_ids, highway)[:, 0, :]
		# gt_emb = self.net.embedder.token_embedding(gt_ids)
		# ukw_mask = gt_ids == self.net.token_embedder.ukw_idx

		# loss = self.criterion(pred_emb, gt_emb, ~ukw_mask)
		topk = self.infere_top_k(val_batch, k=10).tolist()
		topk = [[self.net.token_embedder.i2w[x] for x in lx] for lx in topk]
		gt = [[self.net.token_embedder.i2w[x[0]] for x in lx if x[0] != self.net.token_embedder.ukw_idx] for lx in gt_ids]

		return {'topk': topk, 'gt': gt}

	def test_step(self, val_batch, batch_idx):
		return self.validation_step(self, val_batch, batch_idx)

	def build_match_matrix(self, topk, gt):
		return build_match_matrix(topk, gt)

	def validation_step_end(self, outputs):
		correct = self.build_match_matrix(outputs['topk'], outputs['gt'])
		for name, metric in self.val_metrics.items():
			metric(correct)
			self.log(name, metric, on_step=False, on_epoch=True)
		return outputs

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
