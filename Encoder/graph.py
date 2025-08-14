import torch
from torch_geometric.data import Data
from Encoder.DAT.models import GraphTransformerNet
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import scipy.sparse.linalg
from collections import defaultdict
import torch.nn.functional as F

import argparse, json
import numpy as np
import os
import socket
import time
import random
import glob
import torch.optim as optim
import pickle as pkl


node_features = []
edges = []
edge_features = []
node_index_mapping = {}

with open('Data/graph1.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if parts[0] == 'node':
            original_node_id = int(parts[1])
            node_index = len(node_features)
            node_index_mapping[original_node_id] = node_index
            features = list(map(float, parts[2:]))
            node_features.append(features)
        elif parts[0] == 'edge':
            src = node_index_mapping[int(parts[1])]
            dst = node_index_mapping[int(parts[2])]
            edges.append([src, dst])
            edge_attr = list(map(float, parts[3:]))
            edge_features.append(edge_attr)

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
x = torch.tensor(node_features, dtype=torch.float)
edge_attr = torch.tensor(edge_features, dtype=torch.float)
edge_features_distance = []
for edge in edges:
    source, target = edge
    source_feature = x[source]
    target_feature = x[target]
    distance = torch.norm(source_feature - target_feature, p=2)
    edge_features_distance.append([distance.item()])
edge_attr1 = torch.tensor(edge_features_distance, dtype=torch.float)

neighbors = defaultdict(set)
for source, target in edges:
    neighbors[source].add(target)
    neighbors[target].add(source)
common_friends_ratios = []
for edge in edges:
    source, target = edge
    source_neighbors = neighbors[source]
    target_neighbors = neighbors[target]
    intersection = source_neighbors.intersection(target_neighbors)
    union = source_neighbors.union(target_neighbors)
    ratio = len(intersection) / len(union) if len(union) > 0 else 0
    common_friends_ratios.append([ratio])
edge_attr2 = torch.tensor(common_friends_ratios, dtype=torch.float)

print(edge_index)
print(x)
print(edge_attr)
print(edge_attr1)
print(edge_attr2)
edge_attr = edge_attr+edge_attr1+edge_attr2
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_attr1=edge_attr1, edge_attr2=edge_attr2)

print(data)

def laplacian_positional_encoding(g, pos_enc_dim):
    weight = []
    device = torch.device('cuda:0')
    if g.edge_index.device != device:
        g.edge_index = g.edge_index.to(device)
    num_nodes = g.num_nodes
    edge_index = torch.tensor(g.edge_index, dtype=torch.long)
    g.edge_index = edge_index
    g.features = g.features
    edge_weight = torch.tensor(g.edge_attr, dtype=torch.float)
    edge_weight = edge_weight.view(len(edge_weight), 1)
    if edge_weight.device != device:
        edge_weight = edge_weight.to(device)
    g.edge_weight = edge_weight
    if g.edge_weight.device != device:
        g.edge_weight = g.edge_weight.to(device)
    if g.edge_attr.device != device:
        g.edge_attr = g.edge_attr.to(device)

    laplacian = get_laplacian(edge_index, edge_weight=edge_weight, normalization='sym', num_nodes=num_nodes)
    laplacian_matrix = to_scipy_sparse_matrix(laplacian[0], edge_attr=laplacian[1], num_nodes=num_nodes)

    eigvals, eigvecs = scipy.sparse.linalg.eigsh(laplacian_matrix.asfptype(), k=pos_enc_dim + 1, which='SM')

    eigvecs = eigvecs[:, 1:pos_enc_dim + 1]

    g.lap_pos_enc = torch.from_numpy(eigvecs).float()

    return g

def MyGT_train(g, net):
    if net['lap_pos_enc']:
        print("[!] Adding Laplacian positional encoding.")
        g = laplacian_positional_encoding(g, net['pos_enc_dim'])
    model = GraphTransformerNet(net)
    for param in model.parameters():
        param.requires_grad = False
    device = torch.device('cuda:0')
    model = model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-2,
        weight_decay=1e-2,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5,
                                                     patience=15,
                                                     verbose=True)
    lap_pos_enc = g.lap_pos_enc
    sign_flip = torch.rand(lap_pos_enc.size(1))
    sign_flip[sign_flip >= 0.5] = 1.0;sign_flip[sign_flip < 0.5] = -1.0
    lap_pos_enc =lap_pos_enc * sign_flip.unsqueeze(0)
    g.x = g.x / g.x.sum(1, keepdim=True)
    h, e, edge_index = model.forward(g, lap_pos_enc)
    return h, e, edge_index

#DIFFUSION-AWARE TRANFOEMER
def MyGT_gen(dataset_g):

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--pos_enc_dim', help="Please give a value for pos_enc_dim")
    parser.add_argument('--lap_pos_enc', help="Please give a value for lap_pos_enc")
    parser.add_argument('--wl_pos_enc', help="Please give a value for wl_pos_enc")
    args = parser.parse_args()
    with open("/home/pmy/Encoder/configs/configs1.json") as f:
        config = json.load(f)

    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        dataset = dataset_g

    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    net_params = {}
    net_params_r = {}
    net_params = config['net_params']
    net_params['device'] = 'cuda:0'
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual == 'True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat == 'True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm == 'True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm == 'True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop == 'True' else False
    if args.lap_pos_enc is not None:
        net_params['lap_pos_enc'] = True if args.pos_enc == 'True' else False
    if args.pos_enc_dim is not None:
        net_params['pos_enc_dim'] = int(args.pos_enc_dim)
    if args.wl_pos_enc is not None:
        net_params['wl_pos_enc'] = True if args.pos_enc == 'True' else False

    # graph1
    net_params['node_in_dim'] = dataset.features.shape[1]
    net_params['edge_in_dim'] = 1

    net_params_r = net_params

    h1, e1, edge_index1 = MyGT_train(dataset_g, net_params_r)

    return h1, e1, edge_index1
