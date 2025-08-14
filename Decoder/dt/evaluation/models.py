import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add, scatter_softmax, scatter_max
from torch.nn import Embedding
from torch.utils.data import DataLoader
from torch_geometric.utils.num_nodes import maybe_num_nodes
from collections import deque
from tqdm import tqdm
import random
import numpy as np

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)

EPS = 1e-15

class predict_action(nn.Module):

    def __init__(self, reg_hidden, embed_dim, node_dim, edge_dim, w_scale, avg=False):
        super(predict_action, self).__init__()
        self.T = T
        self.embed_dim = embed_dim
        self.reg_hidden = reg_hidden
        self.avg = avg

        # input node to latent
        self.w_n2l = torch.nn.Parameter(torch.Tensor(node_dim, embed_dim))
        torch.nn.init.normal_(self.w_n2l, mean=0, std=w_scale)

        # input edge to latent
        self.w_e2l = torch.nn.Parameter(torch.Tensor(edge_dim, embed_dim))
        torch.nn.init.normal_(self.w_e2l, mean=0, std=w_scale)

        # linear node conv
        self.p_node_conv = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.p_node_conv, mean=0, std=w_scale)

        # trans node 1
        self.trans_node_1 = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.trans_node_1, mean=0, std=w_scale)

        # trans node 2
        self.trans_node_2 = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.trans_node_2, mean=0, std=w_scale)

        if self.reg_hidden > 0:
            self.h1_weight = torch.nn.Parameter(torch.Tensor(2 * embed_dim, reg_hidden))
            torch.nn.init.normal_(self.h1_weight, mean=0, std=w_scale)
            self.h2_weight = torch.nn.Parameter(torch.Tensor(reg_hidden, 1))
            torch.nn.init.normal_(self.h2_weight, mean=0, std=w_scale)
            self.last_w = self.h2_weight
        else:
            self.h1_weight = torch.nn.Parameter(torch.Tensor(2 * embed_dim, 1))
            torch.nn.init.normal_(self.h1_weight, mean=0, std=w_scale)
            self.last_w = self.h1_weight

        # S2V scatter message passing
        self.scatter_aggr = (scatter_mean if self.avg else scatter_add)

    def forward(self, data):
        data.x = torch.matmul(data.x, self.w_n2l)
        data.x = F.relu(data.x)
        data.edge_attr = torch.matmul(data.edge_attr, self.w_e2l)

        for _ in range(self.T):
            msg_linear = torch.matmul(data.x, self.p_node_conv)
            n2e_linear = msg_linear[data.edge_index[0]]

            edge_rep = torch.add(n2e_linear, data.edge_attr)
            edge_rep = F.relu(edge_rep)

            e2n = self.scatter_aggr(edge_rep, data.edge_index[1], dim=0, dim_size=data.x.size(0))

            data.x = torch.add(torch.matmul(e2n, self.trans_node_1),
                               torch.matmul(data.x, self.trans_node_2))
            data.x = F.relu(data.x)

        y_potential = self.scatter_aggr(data.x, data.batch, dim=0)
        if data.y is not None:
            action_embed = data.x[data.y]

            embed_s_a = torch.cat((action_embed, y_potential), dim=-1)  # ConcatCols

            last_output = embed_s_a
            if self.reg_hidden > 0:
                hidden = torch.matmul(embed_s_a, self.h1_weight)
                last_output = F.relu(hidden)
            q_pred = torch.matmul(last_output, self.last_w)

            return q_pred

        else:
            rep_y = y_potential[data.batch]
            embed_s_a_all = torch.cat((data.x, rep_y), dim=-1)  # ConcatCols

            last_output = embed_s_a_all
            if self.reg_hidden > 0:
                hidden = torch.matmul(embed_s_a_all, self.h1_weight)
                last_output = torch.relu(hidden)

            q_on_all = torch.matmul(last_output, self.last_w)

            return q_on_all