import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
import copy
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class HGFF(nn.Module):
    def __init__(self,
                 n_obs_in=7,
                 n_action=25,
                 n_layers=3,
                 n_features=64,
                 tied_weights=False,
                 n_hid_readout=[], ):

        super().__init__()

        self.n_obs_in = n_obs_in
        self.n_action=n_action
        self.n_layers = n_layers
        self.n_features = n_features
        self.tied_weights = tied_weights
        self.trained_model=None
        self.node_init_embedding_layer = nn.Sequential(
            nn.Linear(n_obs_in, n_features, bias=False),
            nn.ReLU()
        )
        self.type_embedding = nn.Embedding(3, n_obs_in)
        self.edge_embedding_layer = EdgeAndNodeEmbeddingLayer(n_obs_in, n_features)

        if self.tied_weights:
            self.update_node_embedding_layer = UpdateNodeEmbeddingLayer(n_features)
        else:
            self.update_node_embedding_layer = nn.ModuleList(
                [UpdateNodeEmbeddingLayer(n_features) for _ in range(self.n_layers)])
        self.multi_head_attention = MultiHeadAttentionLayer(8, n_features, q_dim=self.n_action)
        self.readout_layer = ReadoutLayer(n_features, n_action, n_hid_readout)

    @torch.no_grad()
    def get_normalisation(self, adj):

        norm = torch.sum((adj != 0), dim=1).unsqueeze(-1)
        norm[norm == 0] = 1
        return norm.float()

    def forward(self, obs):
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        obs.transpose_(-1, -2)

        node_features = obs[:, :, 0:self.n_obs_in]

        # Get graph adj matrix.
        adj = obs[:, :, self.n_obs_in:]

        norm = self.get_normalisation(adj)
        index = node_features[:, :, 0].unsqueeze(-1)
        type_embedding = self.type_embedding(index.long()).reshape(node_features.size(0), node_features.size(1),
                                                                   self.n_obs_in)
        node_features = node_features + type_embedding
        init_node_embeddings = self.node_init_embedding_layer(node_features)
        edge_embeddings = self.edge_embedding_layer(node_features, adj, norm)

        # Initialise embeddings.
        current_node_embeddings = init_node_embeddings

        if self.tied_weights:
            for _ in range(self.n_layers):
                current_node_embeddings = self.update_node_embedding_layer(current_node_embeddings,
                                                                           edge_embeddings,
                                                                           norm,
                                                                           adj)
        else:
            for i in range(self.n_layers):
                current_node_embeddings = self.update_node_embedding_layer[i](current_node_embeddings,
                                                                              edge_embeddings,
                                                                              norm,
                                                                              adj)
        current_node_embeddings = self.multi_head_attention(current_node_embeddings)
        out = self.readout_layer(current_node_embeddings)
        out = out.squeeze()

        return out

    def sample_action(self, obs, epsilon,aciton_size):
        out = self.forward(obs)

        coin = random.random()
        if coin < epsilon:
            return random.randint(0, aciton_size-1)
        else:
            return out.argmax().item()

    def load_models(self,path):

        self.load_state_dict(torch.load(path))
        self.eval()
    def greedy(self,obs):
        out = self.forward(obs)

        return out.argmax().item()

class EdgeAndNodeEmbeddingLayer(nn.Module):

    def __init__(self, n_obs_in, n_features):
        super().__init__()
        self.n_obs_in = n_obs_in
        self.n_features = n_features

        self.edge_embedding_NN = nn.Linear(int(n_obs_in + 1), n_features - 1, bias=False)
        self.edge_feature_NN = nn.Linear(n_features, n_features, bias=False)

    def forward(self, node_features, adj, norm):
        edge_features = torch.cat([adj.unsqueeze(-1),
                                   node_features.unsqueeze(-2).transpose(-2, -3).repeat(1, adj.shape[-2], 1, 1)],
                                  dim=-1)

        edge_features *= (adj.unsqueeze(-1) != 0).float()

        edge_features_unrolled = torch.reshape(edge_features, (
        edge_features.shape[0], edge_features.shape[1] * edge_features.shape[1], edge_features.shape[-1]))
        embedded_edges_unrolled = F.relu(self.edge_embedding_NN(edge_features_unrolled))
        embedded_edges_rolled = torch.reshape(embedded_edges_unrolled,
                                              (adj.shape[0], adj.shape[1], adj.shape[1], self.n_features - 1))
        embedded_edges = embedded_edges_rolled.sum(dim=2) / norm

        edge_embeddings = F.relu(self.edge_feature_NN(torch.cat([embedded_edges, norm / norm.max()], dim=-1)))

        return edge_embeddings


class UpdateNodeEmbeddingLayer(nn.Module):

    def __init__(self, n_features):
        super().__init__()

        self.message_layer = nn.Linear(2 * n_features, n_features, bias=False)
        self.update_layer = nn.Linear(2 * n_features, n_features, bias=False)

    def forward(self, current_node_embeddings, edge_embeddings, norm, adj):
        node_embeddings_aggregated = torch.matmul(adj, current_node_embeddings) / norm

        message = F.relu(self.message_layer(torch.cat([node_embeddings_aggregated, edge_embeddings], dim=-1)))
        new_node_embeddings = F.relu(self.update_layer(torch.cat([current_node_embeddings, message], dim=-1)))

        return new_node_embeddings


class ReadoutLayer(nn.Module):

    def __init__(self, n_features, n_action,n_hid=[], bias_pool=False, bias_readout=True):

        super().__init__()

        self.layer_pooled = nn.Linear(int(n_features), int(n_features), bias=bias_pool)
        self.n_action=n_action

        if type(n_hid) != list:
            n_hid = [n_hid]

        n_hid = [2 * n_features] + n_hid + [1]

        self.layers_readout = []
        for n_in, n_out in list(zip(n_hid, n_hid[1:])):
            layer = nn.Linear(n_in, n_out, bias=bias_readout)
            self.layers_readout.append(layer)

        self.layers_readout = nn.ModuleList(self.layers_readout)

    def forward(self, node_embeddings):

        site_embedding = node_embeddings[:, :self.n_action, :]
        # sensor_embedding = node_embeddings[:, self.n_action:, :]
        f_local = site_embedding
        # tt=node_embeddings[:, :self.n_action, :].sum(dim=1)

        h_pooled = self.layer_pooled(site_embedding.sum(dim=1) / self.n_action)
        f_pooled = h_pooled.repeat(1, 1, self.n_action).view(node_embeddings[:,:self.n_action,:].shape)

        features = F.relu(torch.cat([f_pooled, f_local], dim=-1))

        for i, layer in enumerate(self.layers_readout):
            features = layer(features)
            if i < len(self.layers_readout) - 1:
                features = F.relu(features)
            else:
                out = features

        return out

class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            q_dim

    ):
        super(MultiHeadAttentionLayer, self).__init__(

                MultiHeadAttention(
                    n_heads,
                    q_dim,
                    input_dim=embed_dim,
                    embed_dim=embed_dim,

                )

        )

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            q_dim,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):

        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.q_dim=q_dim
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):

        h = q[:,self.q_dim:,:]
        q_=q[:,:self.q_dim,:]


        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q_.size(1)
        assert q_.size(0) == batch_size
        assert q_.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q_.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

    
        return out
