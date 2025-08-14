import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
import math
import numpy as np

"""
    Graph Transformer Layer with edge features
    
"""

"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func

def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

def imp_exp_attn(implicit_attn, explicit_edge):
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
    return func

def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}
    return func


def exp(field):
    def func(edges):
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func




"""
    Single Attention Head
"""


class GraphMultiHeadAttentionLayerWithEdgeFeatures(MessagePassing):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.1, use_bias=False):
        super(GraphMultiHeadAttentionLayerWithEdgeFeatures, self).__init__(node_dim=0)
        assert out_dim % num_heads == 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.edge_dim = in_dim
        self.head_dim = out_dim

        self.lin_q = nn.Linear(in_dim, out_dim*num_heads, bias=use_bias)
        self.lin_k = nn.Linear(in_dim, out_dim*num_heads, bias=use_bias)
        self.lin_v = nn.Linear(in_dim, out_dim*num_heads, bias=use_bias)
        self.lin_edge = nn.Linear(in_dim, out_dim*num_heads, bias=use_bias)
        self.edge_update_linear = nn.Linear(3*in_dim,out_dim*num_heads, bias=use_bias)

        self.att_dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(out_dim*num_heads, out_dim*num_heads, bias=use_bias)

        self.reset_parameters()

    def reset_parameters(self, use_bias=False):
        nn.init.xavier_uniform_(self.lin_q.weight)
        nn.init.xavier_uniform_(self.lin_k.weight)
        nn.init.xavier_uniform_(self.lin_v.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)
        if use_bias:
            nn.init.zeros_(self.lin_q.bias)
            nn.init.zeros_(self.lin_k.bias)
            nn.init.zeros_(self.lin_v.bias)
            nn.init.zeros_(self.lin_edge.bias)
            nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, edge_index, edge_attr):

        # Transform node features
        q = self.lin_q(x).view(-1, self.num_heads, self.out_dim)
        k = self.lin_k(x).view(-1, self.num_heads, self.out_dim)
        v = self.lin_v(x).view(-1, self.num_heads, self.out_dim)

        # Transform edge features
        edge_attr = self.lin_edge(edge_attr).view(-1, self.num_heads, self.out_dim)

        # Start message passing
        out = self.propagate(edge_index, q=q, k=k, v=v, edge_attr=edge_attr, size=None)

        # Reshape the output
        out = out.view(-1, self.out_dim*self.num_heads)

        # Apply final linear transformation and dropout
        out = self.proj_out(out)

        edge_attr = edge_attr.view(-1, self.out_dim*self.num_heads)

        edge_features_updated = self.update_edge_features(x, edge_index, edge_attr, out)
        h_out = out
        e_out = edge_features_updated
        return h_out, e_out

    def update_edge_features(self, x, edge_index, edge_attr, node_features_updated):
        src, dst = edge_index[0], edge_index[1]
        edge_features_new = torch.cat([x[src], node_features_updated[dst], edge_attr], dim=-1)
        edge_features_new = self.edge_update_linear(edge_features_new)
        return edge_features_new

    def message(self, q_i, k_j, v_j, edge_attr, edge_index, size):
        scores = (q_i * k_j).sum(dim=-1) / math.sqrt(self.head_dim) + (q_i * edge_attr).sum(dim=-1) / math.sqrt(self.head_dim)
        scores = softmax(scores, edge_index[0], num_nodes=size[0])

        scores = self.att_dropout(scores)

        return scores.view(-1, 1, self.num_heads) * (v_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
    

class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        
        self.attention = GraphMultiHeadAttentionLayerWithEdgeFeatures(in_dim, out_dim//num_heads, num_heads, use_bias)
        
        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)
        
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)
        
        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_e_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)
        
    def forward(self, g, h, e):
        h_in1 = h
        e_in1 = e

        h_attn_out, e_attn_out = self.attention(h_in1, g.edge_index, e_in1)
        
        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        if self.residual:
            h = h_in1 + h
            e = e_in1 + e

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        h_in2 = h
        e_in2 = e

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # FFN for e
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h # residual connection       
            e = e_in2 + e # residual connection  

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)             

        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y