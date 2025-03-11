import networkx as nx
import numpy as np
import torch_geometric
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import WikipediaNetwork, WebKB
from torch_geometric.utils import to_networkx

class BallGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_weight:None):
        # what is the shape of inpur x ? - needed [N, in_channels]
        # edge indices shape needed is [2, E]

        #add self_loops to the adjacency matrix, how to give num nodes?
        #edge_index, _ = add_self_loops(edge_index)
        #print(edge_index)
        # linearly transform node feature matrix
        x = self.lin(x)
        #x = torch.index_select(input=x, index=edge_index[0], dim=0)
        # x_ball = torch.cat([torch.index_select(input=x, index=edge_index[0], dim=0), NOTE THAT IT WILL GIVE INDEX OUT OF RANGE ONE OPTION IS TO GO WITH REINDEXING
        #             torch.index_select(input=x, index=edge_index[1], dim=0)],dim=0)
        #compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(0.5)
        deg_inv_sqrt[deg_inv_sqrt==float('inf')] = 0
        #print(deg_inv_sqrt.shape)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # propagating messages
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, norm=norm)
        out = torch.index_select(input=out, index=min(edge_index[0]), dim=0) #NOTE TRICK IS TO PICK MIN EDGE INDEX AS IT WILL CORRESPOND TO THE CENTER NODE OF THE BALL
        # bias
        out += self.bias
        return torch.squeeze(out)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # normalize node features
        return norm.view(-1,1) *x_j


class BallGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = BallGCNConv(in_channels, hidden_channels)
        self.fc = Linear(hidden_channels, out_channels)
    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(self.conv1(x, edge_index, edge_weight))
        return x