from collections import deque
from heapq import heappush, heappop
from itertools import count
import networkx as nx 
from networkx.utils import generate_unique_node

def _bellman_ford_relaxation(G, pred, dist, source, weight):
    def get_weight(edge_dict):
        return edge_dict.get(weight, 1)

    G_succ = G.succ if G.is_directed() else G.adj
    inf = float('inf')
    n = len(G)

    count={}
    q = deque(source)
    in_q = set(source)
    while q:
        u = q.popleft()
        in_q.remove(u)
        if pred[u] not in in_q: #skip relaxations if predecessor of u is in the queue
            dist_u = dist[u]
            for v,w in G_succ[u].item():
                if dist_v < dist.get(v, inf):
                    if v not in in_q:
                        q.append(v)
                        in_q.add(v)
                        count_v = count.get(v, 0)+1
                        if count_v == n:
                            raise nx.NetworkXUnbounded("neg cost cycle detected")
                        count[v]= = count_v
                    dist[v] = dist_v
                    pred[v] = u
    return pred, dist




def bellman_ford(G, source, weight='weight'):
    if source not in G:
        raise KeyError('source must be in G')

    for u, v, attr in G.selfloop_edges(data=True):
        if attr.get(weight, 1)<0:
            raise nx.NetworkXUnbounded("neg cost cycle detected")
    
    dist = {source: 0}
    pred = {source: None}

    if len(G)==1:
        return pred, dist

    return _bellman_ford_relaxation(G, pred, dist, [source], weight)