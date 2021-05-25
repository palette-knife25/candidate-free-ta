import torch
from hydra.utils import instantiate
from torch.nn import functional as F

from models.fixed_enricher import FixedTokenizer
from options import ExperimentConfig
from .base_system import BaseSystem


class FixedTESystem(BaseSystem):
	def __init__(self, opt: ExperimentConfig, tokenizer: FixedTokenizer):
		super().__init__(opt, tokenizer)
		self.net = instantiate(opt.net, tokenizer)

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

		topk = self.infere_top_k(val_batch, k=10).tolist()
		topk = [[self.net.token_embedder.i2w[x] for x in lx] for lx in topk]
		gt = [[self.net.token_embedder.i2w[x[0]] for x in lx if x[0] != self.net.token_embedder.ukw_idx] for lx in gt_ids]

		return {'topk': topk, 'gt': gt}

