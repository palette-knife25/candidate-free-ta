import torch
from torch import nn
from transformers import BertModel


class KBertEnricher(nn.Module):
	def __init__(self, base_model, type_embedding_max):
		super().__init__()
		pretrained_bert = BertModel.from_pretrained(base_model)
		bert_config = pretrained_bert.config
		input_embedding = pretrained_bert.embeddings.word_embeddings
		input_embedding.weight.requires_grad = False
		self.embedding = TaxoEmbedding(
			input_embedding,
			bert_config.hidden_size,
			type_embedding_max,
			bert_config.max_position_embeddings,
			bert_config.hidden_dropout_prob
		)
		self.encoder = pretrained_bert.encoder
		self.head = nn.Linear(bert_config.hidden_size, bert_config.vocab_size)

	def forward(self, token_ids, type_ids, synset_ids, highway):
		visibility_mask = build_visibility_mask(synset_ids, highway)  # b x seq_len x seq_len
		not_padding_mask = (token_ids != self.embedding.token_embedding.padding_idx)  # b x seq_len
		# combine masks and make broadcastable to all heads
		attn_mask = visibility_mask[:, None, :, :] & not_padding_mask[:, None, None, :]  # b x 1 x seq_len x seq_len
		embedding_output = self.embedding(token_ids, type_ids)
		hidden_state = self.encoder(
			embedding_output,
			attention_mask=attn_mask.float()).last_hidden_state
		log_probs = torch.log_softmax(self.head(hidden_state), dim=-1)
		return log_probs  # b x seq_len x vocab_size


class TaxoEmbedding(nn.Module):
	def __init__(self, token_embedding, hidden_size, type_embedding_max, pos_embedding_max, dropout):
		super().__init__()
		self.token_embedding = token_embedding
		self.type_embedding = nn.Embedding(type_embedding_max, hidden_size)
		self.position_embedding = nn.Embedding(pos_embedding_max, hidden_size)
		self.layer_norm = nn.LayerNorm(hidden_size)
		self.dropout = nn.Dropout(dropout)
		self.register_buffer("position_ids", torch.arange(pos_embedding_max).expand(1, -1))

	def forward(self, token_ids, type_ids):
		position_ids = self.position_ids[:, :token_ids.shape[-1]]
		embeddings = self.token_embedding(token_ids) + self.type_embedding(type_ids) + self.position_embedding(
			position_ids)
		embeddings = self.dropout(self.layer_norm(embeddings))
		return embeddings


def build_visibility_mask(synset_ids, highway):
	visibility_mask = synset_ids.unsqueeze(-1) == synset_ids.unsqueeze(-2)
	visibility_mask |= highway.unsqueeze(-1) & highway.unsqueeze(-2)
	return visibility_mask
