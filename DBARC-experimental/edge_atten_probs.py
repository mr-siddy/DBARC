import os
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Edge_atten(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=1, concat_heads=True, alpha=0.2):
        super().__init__()
        self.num_heads=num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert out_channels % num_heads==0, "number of output channels must be multiple of count of heads"
            out_channels = out_channels // num_heads

        self.linear = nn.Linear(in_channels, out_channels*num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2*out_channels))
        self.leakyrelu = nn.LeakyReLU(alpha)

        #xavier uniform initialization
        nn.init.xavier_uniform_(self.linear.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
    
    def forward(self, node_feats, edge_index):
        node_feats = torch.unsqueeze(node_feats, dim=0)
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)
        node_feats = self.linear(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)
        node_feats_flat = node_feats.view(batch_size*num_nodes, self.num_heads, -1)
        edge_indices_row = edge_index[0]
        edge_indices_col = edge_index[1]
        a_input = torch.cat([
            torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
            torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)
        ], dim=-1)
        attn_logits = torch.einsum('bhc, hc->bh', a_input, self.a)
        attn_logits = self.leakyrelu(attn_logits)
        attn_probs = F.softmax(attn_logits, dim=-2)
        return attn_probs