import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch.nn import Linear



"""
    Graph Transformer with edge features

"""
from Encoder.DAT.layers import GraphTransformerLayer
from Encoder.DAT.layers import MLPReadout


class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        node_in_dim = net_params['node_in_dim']
        edge_in_dim = net_params['edge_in_dim']
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        max_wl_role_index = 37

        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)

        self.gat_layer = GATConv(in_channels=node_in_dim, out_channels=hidden_dim, heads=1, concat=True)

        if self.edge_feat:
            self.edge_linear = Linear(in_features=edge_in_dim, out_features=hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                           self.layer_norm, self.batch_norm, self.residual) for _ in
                                     range(n_layers - 1)])
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))
        self.MLP_layer = MLPReadout(out_dim, 1)

    def forward(self, g, h_lap_pos_enc=None):

        device = torch.device('cuda:0')
        if g.edge_index.device != device:
            g.edge_index = g.edge_index.to(device)
        if g.edge_weight.device != device:
            g.edge_weight = g.edge_weight.to(device)
        if g.lap_pos_enc.device != device:
            g.lap_pos_enc = g.lap_pos_enc.to(device)
        g.features = torch.tensor(g.features)
        if g.features.device != device:
            g.features = g.features.to(device)
        g.features = g.features.to(dtype=torch.float32)
        g.lap_pos_enc = g.lap_pos_enc.to(dtype=torch.float64)
        h = self.gat_layer(x=g.features, edge_index=g.edge_index)
        h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            if h_lap_pos_enc.device != device:
                h_lap_pos_enc = h_lap_pos_enc.to(device)
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc

        if not self.edge_feat:
            e = torch.ones(e.size(0), 1).to(self.device)
        e = self.edge_linear(g.edge_weight)

        for conv in self.layers:
            h, e = conv(g, h, e)
            edge_index = g.edge_index
        h = h[:,-1]
        return h, e, edge_index

