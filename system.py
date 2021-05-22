from options import ExperimentConfig
import pytorch_lightning as pl
import torchmetrics

import torch
from torch import nn
from torch.nn import functional as F
from models import KBertEnricher
from itertools import chain
from metrics import PrecisionK, MeanAveragePrecision
from metrics.mrr import build_match_matrix, MeanReciprocalRank


class CandidateFreeTE(pl.LightningModule):
	def __init__(self, opt: ExperimentConfig, tokenizer):
		super().__init__()
		self.opt = opt
		self.tokenizer = tokenizer
		self.mask_token_id = self.tokenizer.mask_token_id
		self.net = KBertEnricher(opt.base_model, opt.type_embedding_max)
		self.val_metrics = {"ValidationMRR": MeanReciprocalRank(),
							"ValidationPrecision@10": PrecisionK(),
							"ValidationMAP": MeanAveragePrecision()}

	def forward(self, token_ids, type_ids, synset_ids, highway):
		return self.net(token_ids, type_ids, synset_ids, highway)

	@torch.no_grad()
	def infere_top_k(self, batch, k, insert_position=0):
		if len(batch) == 2:
			batch = batch[0]
		all_token_ids = []
		all_scores = []
		for n_tokens in range(1, self.opt.max_tokens_lemma + 1):
			token_ids_m, type_ids_m, synset_ids_m, highway_m = self.insert_mask_tokens(*batch, n_tokens, insert_position)
			batch_size = token_ids_m.shape[0]
			log_probs = self.forward(token_ids_m, type_ids_m, synset_ids_m, highway_m)
			topk_scores, topk_token_ids = torch.topk(log_probs[:, insert_position, :], k)
			topk_token_ids = topk_token_ids.unsqueeze(-1)
			for mask_index in range(insert_position + 1, insert_position + n_tokens):
				next_topk_scores = []
				next_topk_token_ids = []
				for i in range(k):
					token_ids_m[:, insert_position:mask_index] = topk_token_ids[:, i].view(batch_size, -1)
					log_probs = self.forward(token_ids_m, type_ids_m, synset_ids_m, highway_m)
					best_scores, best_token_ids = torch.topk(log_probs[:, mask_index, :], k)
					next_topk_scores.append((topk_scores[:, i].unsqueeze(-1).expand_as(best_scores) * best_scores))
					next_topk_token_ids.append(torch.cat([topk_token_ids[:, i].unsqueeze(1).expand_as(topk_token_ids),
														  best_token_ids.unsqueeze(-1)], dim=-1))
				next_topk_scores = torch.cat(next_topk_scores, dim=1)
				next_topk_token_ids = torch.cat(next_topk_token_ids, dim=1)
				topk_scores, topk_indices = torch.topk(next_topk_scores, k, dim=-1)
				topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, next_topk_token_ids.shape[-1])
				topk_token_ids = torch.gather(next_topk_token_ids, 1, topk_indices)
			all_token_ids.append(topk_token_ids.tolist())
			all_scores.append(topk_scores)
		all_scores = torch.cat(all_scores, dim=1)
		# len(num_tokens) x batch_size x k -> batch_size x len(num_tokens) * k
		all_token_ids = [list(chain(*pred_list)) for pred_list in zip(*all_token_ids)]
		topk_scores, topk_indices = torch.topk(all_scores, k, dim=1)
		topk_lemmas = [[self.tokenizer.decode(all_token_ids[batch_entry][index]) for index in topk_indices[batch_entry]]
					   for batch_entry in range(topk_indices.shape[0])]
		return topk_lemmas, topk_scores

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
		topk = self.infere_top_k((token_ids, type_ids, synset_ids, highway), self.opt.top_k)[0]

		return {'topk': topk, 'gt': [self.tokenizer.batch_decode(gt) for gt in gt_ids]}

	def test_step(self, val_batch, batch_idx):
		return self.validation_step(self, val_batch, batch_idx)

	def validation_step_end(self, outputs):
		correct = build_match_matrix(outputs['topk'], outputs['gt'])
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
