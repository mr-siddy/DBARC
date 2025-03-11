from collections import deque
from heapq import heappop, heappush
from itertools import count
from networkx import nx
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors

def _weight_function(G, weight):
    if callabe(weight):
        return weight
    if G.is_multigraph():
        return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values)
    return lambda u, v, data: data.get(weight, 1)

def _bellman_ford(G, source, weight, pred=None, paths=None, dist=None, target=None, heuristic=True):
    if pred is None:
        pred = {v: [] for v in source}
    if dist is None:
        dist = {v:0 for v in source}
    
    negative_cycle_found = _bellman_ford_relaxation(G, source, weight, pred, dist, heuristic,)
    if negative_cycle_found is not None:
        raise ValueError
    if paths is not None:
        sources = set(source)
        dsts = [target] if target is not None else pred
        for dst in dsts:
            gen = _build_paths_from_predecessors(sources, dst, pred)
            paths[dst] = next(gen)
    return dist

def _bellman_ford_relaxation(G, sources, weight, pred, dist=None, heuristic=True):
    for s in sources:
        if s not in G:
            raise ValueError
    if pred is None:
        pred = {v: [] for v in sources}
    if dist is None:
        dist = {v: 0 for v in sources}
    
    nonexistent_edges = (None, None)
    pred_edge = {v: None for v in sources}
    recent_update = {v: nonexistent_edge for v in sources}
    G_succ = G._adj #works both for directed and undirected graphs
    inf = float("inf")
    n = len(G)

    count={}
    q = deque(sources)
    in_q = set(sources)

    while q:
        u = q.popleft()
        in_q.remove(u)
        #skip relaxation if any of the predecessors of u is in the queue
        if all(pred_u not in in_q for pred_u in pred[u]):
            dist_u = dist[u]
            for v, e in G_succ[u].items():
                dist_v = dist_u + weight(u,v,e)
                if dist_v < dist.get(v, inf):
                    if heuristic:
                        if v in recent_update[u]:
                            #neg cycle
                            pred[v].append(u)
                            return v
                        if v in pred_edge and pred_edge[v] == u:
                            recent_update[v] == recent_update[u]
                        else:
                            recent_update[v] = (u, v)
                    if v not in in_q:
                        q.append(v)
                        in_q.add(v)
                        count_v = count.get(v, 0)+1
                        if count_v==n:
                            #neg cycle
                            return v

                        count[v] = count_v
                    dist[v] = dist_v
                    pred[v] = [u]
                    pred_edge[v] = u 
                
                elif dist.get(v) is not None and dist_v == dist.get(v):
                    pred[v].append(u)
    return None


