import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import numpy as np
import time
import torch_geometric
from torch_gemetric.datasets import WikipediaNetwork, WebKB, Planetoid
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import to_networkx
from edge_atten_probs import Edge_atten 
from ball_gcn import BallGCNConv, BallGCN
from k_ball import init_ball

dataset = WikipediaNetwork(root="/home/siddy/META/data", name='chameleon')
data = dataset[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act

    def forward(self, x):
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        out = self.act(hidden)
        return out

edge_attn = Edge_atten(in_channels=data.x.size(1), out_channels=64)
ball_gcn = BallGCN(in_channels=data.x.size(1), hidden_channels=64, out_channels=dataset.num_classes)
msg = Linear(in_features=data.x.size(1), out_features=dataset.num_classes, dropout=0.2, act=F.ReLU, use_bias=False)

G = to_
edge_dict = init_ball(radius=2, graph=G)

optimizer = torch.optim.Adam(set(edge_attn.parameters())| set(msg.parameters())| set(ball_gcn.parameters()), lr=0.001, weight_decay=5e-4)

def train():
    edge_attn.train()
    msg.train()
    ball_gcn.train()
    feats =[]
    for node, ball in edge_dict.items():
        if len(ball)<1:
            index_feat = msg(data.x[node])
            feats.append(index_feat)
        else:
            edge_list = torch.permute(torch.tensor(ball, dtype=torch.long), (1,0))
            attn_probs = edge_attn(data.x, edge_list)
            out = ball_gcn(data.x, edge_list, attn_probs)
            feats.append(out)
    feat = torch.stack(feats, -2)
    print(feat.shape)
    loss = F.cross_entropy(feat[data.train_mask[:,0]], data.y[data.train_mask[:,0]])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test():
    edge_attn.eval()
    msg.eval()
    ball_gcn.eval()
    feats = []
    for node, ball in edge_dict.items():
        if len(ball)<1:
            index_feat = msg(data.x[node])
            feats.append(index_feat)
        else:
            edge_list = torch.permute(torch.tensor(ball, dtype=torch.long), (1,0))
            attn_probs = edge_attn(data.x, edge_list)
            out = ball_gcn(data.x,edge_list, attn_probs).argmax(dim=-1)
            feats.append(out)
    feat = torch.stack(feats, -1)
    print(feat.shape)
    accs=[]
    for mask in [data.train_mask[:,0], data.val_mask[:,0], data.test_mask[:,0]]:
        accs.append(int((feat[mask] == data.y[mask]).sum())/int(mask.sum()))
    return accs


if __name__ == '__main__':
    epochs=100
    best_val_acc = final_test_acc = 0
    times = []
    for epoch in range(1, epochs+1):
        start = time.time()
        loss= train()
        wandb.log({"Loss":loss})
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        print(f"Epoch:{epoch}, Loss:{loss}, Train:{train_acc}, Val:{val_acc}, Test:{test_acc}")
        wandb.log({"train_acc":train_acc})
        wandb.log({"test_acc":test_acc})
        wandb.log({"val_acc":val_acc})
        times.append(time.time()-start)
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")