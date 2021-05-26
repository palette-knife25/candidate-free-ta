import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))  # W.shape: (in_features, out_features)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, attn_mask):
        # h.shape: (bs, seq_len, in_features)
        # attn_mask.shape: (bs, seq_len, seq_len)

        Wh = torch.matmul(h, self.W)  # Wh.shape: (bs, seq_len, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)  # a_input.shape: (bs, seq_len, seq_len, 2*out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # e.shape: (bs, seq_len, seq_len)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(attn_mask > 0, e, zero_vec)  # attn.shape: (bs, seq_len, seq_len)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # h_prime.shape: (bs, seq_len, out_features)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[1]  # number of nodes (a.k.a. seq_len)
        bs = Wh.size()[0]  # batch size

        # Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)  # Wh_repeated_in_chunks.shape: (bs, N * N, out_features)
        # Wh_repeated_alternating = Wh.repeat(1, N, 1)  # Wh_repeated_alternating.shape: (bs, N * N, out_features)

        all_combinations_matrix = torch.cat([Wh.repeat_interleave(N, dim=1), Wh.repeat(1, N, 1)], dim=2)  # all_combinations_matrix.shape: (bs, N * N, 2 * out_features)

        return all_combinations_matrix.view(bs, N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.1, nheads=4, alpha=1e-2):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(nhid*nheads, nclass, dropout=dropout, alpha=alpha, concat=False)  # replaced in order to fit in memory
        self.out_layer = nn.Linear(nhid * nheads, nclass)

    def forward(self, x, attn_mask):
        # x.shape: (bs, seq_len, nfeat)
        # attn_mask.shape: (bs, seq_len, seq_len)

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, attn_mask) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        #x = F.elu(self.out_att(x, attn_mask))  # replaced in order to fit in memory
        x = F.elu(self.out_layer(x))

        # final x.shape: (bs, seq_len, nclass)
        return F.log_softmax(x, dim=-1)


class KBertGATEnricher(nn.Module):
    def __init__(self, base_model, type_embedding_max, gat_n_heads, gat_hidden_size, bert_encoder=True):
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

        self.bert_encoder=bert_encoder
        if self.bert_encoder:
            self.encoder = pretrained_bert.encoder
        self.gat = GAT(nfeat=bert_config.hidden_size, nhid=gat_hidden_size, nclass=bert_config.vocab_size, dropout=bert_config.hidden_dropout_prob, nheads=gat_n_heads)

    def forward(self, token_ids, type_ids, synset_ids, highway):
        visibility_mask = build_visibility_mask(synset_ids, highway)  # b x seq_len x seq_len
        not_padding_mask = (token_ids != self.embedding.token_embedding.padding_idx)  # b x seq_len
        # combine masks and make broadcastable to all heads
        attn_mask = visibility_mask[:, None, :, :] & not_padding_mask[:, None, None, :]  # b x 1 x seq_len x seq_len
        embedding_output = self.embedding(token_ids, type_ids)

        if self.bert_encoder:
            hidden_state = self.encoder(
                embedding_output,
                attention_mask=attn_mask.float()).last_hidden_state

            graph_visibility_mask = build_graph_mask(type_ids, synset_ids, highway)
            graph_attn_mask = graph_visibility_mask[:, None, :, :] & not_padding_mask[:, None, None, :]  # b x 1 x seq_len x seq_len
            log_probs = self.gat(hidden_state, attn_mask=graph_attn_mask.squeeze(1).float())
        else:
            graph_visibility_mask = build_graph_mask(type_ids, synset_ids, highway)
            graph_attn_mask = graph_visibility_mask[:, None, :, :] & not_padding_mask[:, None, None, :]  # b x 1 x seq_len x seq_len
            log_probs = self.gat(embedding_output, attn_mask=graph_attn_mask.squeeze(1).float())

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

def build_graph_mask(type_ids, synset_ids, highway):
    # The assumed graph rules are:
    # everyone see each other inside one synset
    # only highways see other highways, with following rules (see Andreea's scheme for reference):
    #   type 1 and type 0 see each other
    #   type 1 and type 2 see each other
    #   type 1 and type 5 see each other
    #   type 3 and type 6 see each other
    #   type 3 and type 4 see each other
    #   type 3 and type 0 see each other

    visibility_mask = synset_ids.unsqueeze(-1) == synset_ids.unsqueeze(-2)

    visibility_mask |= ((type_ids == 1) & highway).unsqueeze(-1) & ((type_ids == 0) & highway).unsqueeze(-2)
    visibility_mask |= ((type_ids == 1) & highway).unsqueeze(-1) & ((type_ids == 2) & highway).unsqueeze(-2)
    visibility_mask |= ((type_ids == 1) & highway).unsqueeze(-1) & ((type_ids == 5) & highway).unsqueeze(-2)
    visibility_mask |= ((type_ids == 3) & highway).unsqueeze(-1) & ((type_ids == 6) & highway).unsqueeze(-2)
    visibility_mask |= ((type_ids == 3) & highway).unsqueeze(-1) & ((type_ids == 4) & highway).unsqueeze(-2)
    visibility_mask |= ((type_ids == 3) & highway).unsqueeze(-1) & ((type_ids == 0) & highway).unsqueeze(-2)

    return visibility_mask