from types import SimpleNamespace

import gensim.downloader
import torch
from nltk.tokenize import RegexpTokenizer
from torch import nn

from models.kbert_enricher import TaxoEmbedding


class FixedTokenizer(nn.Module):
    def __init__(self, w2v_model="glove-wiki-gigaword-300"):
        super().__init__()
        self.padding_idx = 0
        self.ukw_idx = 1
        self.mask_token_id = 2
        self.reserved = 3
        self.tokenizer = RegexpTokenizer(r'[a-z]+')
        w2v = gensim.downloader.load(w2v_model)
        vectors = torch.tensor(w2v.vectors)
        self.vocab_size, self.emb_size = vectors.shape[0] + self.reserved, vectors.shape[1]
        vectors = torch.cat([vectors.new_zeros((self.reserved, self.emb_size)), vectors], dim=0)
        self.embedding = nn.Embedding.from_pretrained(vectors, freeze=True, padding_idx=self.padding_idx)
        self.w2i = {w: i + self.reserved for (i, w) in enumerate(w2v.vocab.keys())}
        self.i2w = {v: k for (k, v) in self.w2i.items()}
        self.i2w.update({0: '[PAD]', 1: '[UKW]', 2: '[MASK]'})

    def get_word_idx(self, word):
        if word in self.w2i:
            return self.w2i[word]
        else:
            return self.ukw_idx

    def tokenize(self, lemma, single=False, *args, **kwargs):
        lemma = lemma[0]
        tokens = self.tokenizer.tokenize(lemma.lower())
        if single:
            word = '-'.join(tokens)
            return SimpleNamespace(input_ids=[self.get_word_idx(word)])
        else:
            return SimpleNamespace(input_ids=[self.get_word_idx(word) for word in tokens])

    def batch_encode_plus(self, lemmas, *args, **kwargs):  # always single
        return SimpleNamespace(input_ids=[[self.get_word_idx('-'.join(self.tokenizer.tokenize(lemma.lower())))]
                                          for lemma in lemmas])

    def convert_ids_to_tokens(self, ids):
        return [self.i2w[idx] for idx in ids.tolist()]

    def forward(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def get_embedding(self, ids):
        return self.embedding(ids)


class FixedEnricher(nn.Module):
    def __init__(self, token_embedder: FixedTokenizer, type_embedding_max=8, num_layers=4):
        super().__init__()
        self.padding_idx = token_embedder.padding_idx
        hidden_size = token_embedder.emb_size
        self.token_embedder = token_embedder
        self.embedder = TokenTypeEmbedding(token_embedder.get_embedding, hidden_size, type_embedding_max,
                                           # pos_embedding_max=512,
                                           dropout=0.1)
        self.encoder_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(hidden_size, 4) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_size, token_embedder.emb_size)

    def forward(self, token_ids, type_ids, synset_ids, highway):
        valid_mask = (token_ids != self.padding_idx)
        embedding_output = self.embedder(token_ids, type_ids)
        hidden_state = embedding_output.transpose(0, 1)
        for encoder in self.encoder_layers:
            hidden_state = encoder(hidden_state, src_key_padding_mask=~valid_mask)
        hidden_state = hidden_state.transpose(0, 1)
        return self.head(hidden_state)


class TokenTypeEmbedding(nn.Module):
    def __init__(self, token_embedding, hidden_size, type_embedding_max, dropout=0.1):
        super().__init__()
        self.token_embedding = token_embedding
        self.type_embedding = nn.Embedding(type_embedding_max, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids, type_ids):
        embeddings = self.token_embedding(token_ids) + self.type_embedding(type_ids)
        embeddings = self.dropout(self.layer_norm(embeddings))
        return embeddings
