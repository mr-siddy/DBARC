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

class ATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=1, concat_heads=True, alpha=0.2): #neg slope of leakyReLU activation function
        super().__init__()
        self.num_heads=num_heads
        self.concat_heads= concat_heads
        if self.concat_heads:
            assert out_channels % num_heads==0, "number of output channels must be multiple of count of heads"
            out_channes = out_channels // num_heads
        
        #submodules and parameters needed in the layer
        self.linear = nn.Linear(in_channels, out_channels*num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2*out_channels)) #one per head
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        #xavier_uniform initialization
        nn.init.xavier_uniform_(self.linear.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
    
    def forward(self, node_feats, adj_matrix, print_attn_probs=False):
        """Forward
        Inputs:
        node_feats= [batch_size, in_channels]
        adj_matrix = [batch_size, num_nodes, num_nodes]
        print_attn_probes= for debugging
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)

        node_feats = self.linear(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)
        # We need to calculate the attention logits for every edge in the adjacency matrix
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        print(node_feats.shape)
        edges = adj_matrix.nonzero(as_tuple=False)#returns indices where adjacency matrix is not 0--> edges
        node_feats_flat = node_feats.view(batch_size*num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:,0] * num_nodes + edges[:,1]
        edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]
        a_input = torch.cat([
            torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
            torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)
        ], dim=-1) #dim -1 forces to concat on feature side, [edge_row/col, attn heads, features]

        #calculate attention MLP output independent of each head
        attn_logits = torch.einsum('bhc, hc->bh', a_input, self.a)
        attn_logits = self.leakyrelu(attn_logits)

        #map list of attn values back into the a matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape+(self.num_heads,)).fill_(-9e15)
        attn_matrix[adj_matrix[...,None].repeat(1,1,1,self.num_heads)==1] = attn_logits.reshape(-1)
        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("attn probs\n", attn_probs.permute(0,3,1,2))
        
        node_feats = torch.einsum('bijh, bjhc->bihc', attn_probs, node_feats)

        if self.concat_heads: #it was good to see they didn't use torch.cat here
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)
        return node_feats, attn_probs