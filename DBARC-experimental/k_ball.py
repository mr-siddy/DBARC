import torch
import networkx as nx
from torch import Tensor
from torch_geometric.utils.mask import index_to_mask
from torch_geometric.utils.num_nodes import maybe_num_nodes


def init_ball(radius: int, graph):
    edge_dict = {}
    for index, node in enumerate(graph.nodes()):
        paths = nx.single_source_shortest_path(graph, node, radius)
        if index not in edge_dict:
            edge_dict[index] = []
        for key, value in paths.items():
            if len(value) == 2:
                edge_dict[index].append(value)
            elif len(value)==3:
                edge_dict[index].append(value[1:])
    return edge_dict

