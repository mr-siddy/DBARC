def ball_dist(edge_list):
    edge_list_nx = list(tuple(i) for x, i in enumerate(edge_list.t().numpy()))
    g_ball = nx.Graph()
    g_ball.add_edges_from(edge_list_nx)
    orc= OllivierRicci(g_ball, alpha=0.5)
    orc.compute_ricci_curvature()
    G_orc = orc.G.copy()
    ricci_curvatures = nx.get_edge_attributes(G_orc, "ricciCurvature")
    #print(ricci_curvatures)
    dist_ball = {}
    for key, value in ricci_curvatures.items():
        #print(key)
        dist_ball[key] = [] # keys are the edges
        x, y = feat[key[0]], feat[key[1]]
        euclidean_dist = dist_e(x, y)
        poincare_dist = dist_p(x,y)
        sphere_dist = dist_s(x, y)
        dist_ = euclidean_dist+poincare_dist+sphere_dist
        #print(dist_)
        dist_ = round(float(dist_.detach().cpu().numpy()),2)
    # print(dist_ball[key])
        dist_ball[key].append(value)
        dist_ball[key].append(dist_)
    # print(dist_ball[key])
        # dist_ball[key].append(dist_)
    return dist_ball


def ball_rewiring(dist_ball):
    weighted_edge_list = []
    for edge, values in dist_ball.items():
        weighted_edge_list.append(edge+(values[1],))
    g_rewire = nx.Graph()
    g_rewire.add_weighted_edges_from(weighted_edge_list)
    sp_dict = {}
    for index, node in enumerate(g_rewire.nodes()):
        length, path = nx.single_source_bellman_ford(g_rewire, node, weight='weight')
        if index not in sp_dict:
             sp_dict[index] = []
        sp_dict[index].append(length)
        #sp_dict[index].append(path)
    return sp_dict #shortest path dictionary

def star_rewire(dist_ball):
  dist_ = []
  for edge, value in dist_ball.items():
    dist_.append(value[1])
  rewired_edges = []
  for edge, value in dist_ball.items():
    if value[1] <= np.mean(dist_):
      u,v = edge
      rewired_edges.append([u,v])
  return rewired_edges


def rewired_ball(sp_dict):
    rewired_edges = []
    for index, neighbours in sp_dict.items():
        radius = []
        for node, dist in neighbours[0].items():
            radius.append(dist)
        #print(np.mean(radius))
        for node, dist in neighbours[0].items():
            if dist <= np.mean(radius):
                rewired_edges.append([index, node])
    return rewired_edges