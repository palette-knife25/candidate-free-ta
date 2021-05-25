from itertools import chain

import torch
from hydra.utils import instantiate
from torch.nn import functional as F

from options import ExperimentConfig
from systems.base_system import BaseSystem


class CandidateFreeTE(BaseSystem):
	def __init__(self, opt: ExperimentConfig, tokenizer):
		super().__init__(opt, tokenizer)
		self.opt = opt
		self.net = instantiate(opt.net)

	@torch.no_grad()
	def infere_top_k(self, batch, k, insert_position=0):
		if len(batch) == 2:
			batch = batch[0]
		all_token_ids = []
		all_scores = []
		for n_tokens in range(1, self.opt.max_tokens_lemma + 1):
			token_ids_m, type_ids_m, synset_ids_m, highway_m = self.insert_mask_tokens(*batch, n_tokens, insert_position)
			batch_size = token_ids_m.shape[0]
			probs = self.forward(token_ids_m, type_ids_m, synset_ids_m, highway_m).exp()
			topk_scores, topk_token_ids = torch.topk(probs[:, insert_position, :], k)
			topk_token_ids = topk_token_ids.unsqueeze(-1)
			for mask_index in range(insert_position + 1, insert_position + n_tokens):
				next_topk_scores = []
				next_topk_token_ids = []
				for i in range(k):
					token_ids_m[:, insert_position:mask_index] = topk_token_ids[:, i].view(batch_size, -1)
					probs = self.forward(token_ids_m, type_ids_m, synset_ids_m, highway_m).exp()
					best_scores, best_token_ids = torch.topk(probs[:, mask_index, :], k)
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

