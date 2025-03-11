import torch
import networkx as nx 
import math
import numpy as np
import matplotlib.pyplot as plt

from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
    
import community.community_louvain as community_louvain

from sklearn import preprocessing, metrics


def show_results(G, curvature: str):

    #print and check for 5 edges
    for n1,n2 in list(G.edges())[:5]:
        print("Ricci curvature of edge (%s,%s) is %f" % (n1 ,n2, G[n1][n2][curvature]))

    # Plot the histogram of Ricci curvatures
    plt.subplot(2, 1, 1)
    ricci_curvtures = nx.get_edge_attributes(G, curvature).values()
    plt.hist(ricci_curvtures,bins=20)
    plt.xlabel('Ricci curvature')
    plt.title("Histogram of Ricci Curvatures")

    # Plot the histogram of edge weights
    plt.subplot(2, 1, 2)
    weights = nx.get_edge_attributes(G, "weight").values()
    plt.hist(weights,bins=20)
    plt.xlabel('Edge weight')
    plt.title("Histogram of Edge weights")

    plt.tight_layout()


class DiscreteRicciCurv():
    def __init__(self, G):
        self.G = G
        for (n1, n2, d) in self.G.edges:
            d.clear()
        
    def ollivierricci(self):
        orc = OllivierRicci(self.G, alpha=0.5, verbose="TRACE")
        orc.compute_ricci_curvature()
        G_orc = orc.G.copy()
        return G_orc
    
    def formanricci(self):
        frc = FormanRicci(G, verbose="TRACE")
        frc.compute_ricci_curvature()
        G_frc = frc.G.copy()
        return G_frc