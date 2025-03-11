import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from math_utils import acos, arcosh

class ProductDistance():
    def __init__(self, x, y, k, atten_prob):
        self.x = torch.tensor(x)
        #print(x.dtype)
        self.y = torch.tensor(y)
        self.k = torch.tensor(k)
        self.atten_prob = atten_prob
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
    
    def poincaresqdist(self):
        if len(self.x) != len(self.y):
            raise ValueError("Input vectors must have the same dimensions")
        
        K = 1./ torch.sqrt(- self.k)
        prod = self.atten_prob**2
        #dist = torch.clamp(1. - ((2*self.k*(prod))/((1+self.k*(torch.norm(self.x, p=2.0, dtype=torch.float)**2))*(1+self.k*(torch.norm(self.y, p=2.0, dtype=torch.float)**2)))), min=1.0 + self.eps[self.x.dtype])
        dist = 1. - ((2*self.k*(prod))/((1+self.k*(torch.norm(self.x, p=2.0, dtype=torch.float)**2))*(1+self.k*(torch.norm(self.y, p=2.0, dtype=torch.float)**2))))
        sqdist = (K * arcosh(dist))**2
        return sqdist

    def hyperspheresqdist(self):
        if len(self.x) != len(self.y):
            raise ValueError("Input vectors must have the same dimensions")
        
        K = 1./ torch.sqrt(self.k)
        prod = self.atten_prob**2
        #dist = torch.clamp(1. - ((2*self.k*(prod))/((1+self.k*(torch.norm(self.x, p=2, dtype=torch.float)**2))*(1+self.k*(torch.norm(self.y, p=2, dtype=torch.float)**2)))), min=1.0 + self.eps[self.x.dtype])
        dist = 1. - ((2*self.k*(prod))/((1+self.k*(torch.norm(self.x, p=2, dtype=torch.float)**2))*(1+self.k*(torch.norm(self.y, p=2, dtype=torch.float)**2))))     
        sqdist = (K * acos(dist))**2
        return sqdist

    def euclideansqdist(self):
        if len(self.x) != len(self.y):
            raise ValueError("Input vectors must have the same dimensions")

        dist = torch.norm(self.x - self.y, p=2, dtype=torch.float)
        sqdist = dist**2
        return sqdist

    def productdistance(self):
        poincare_dist = self.poincaresqdist()
        hypersphere_dist = self.hyperspheresqdist()
        euclidean_dist = self.euclideansqdist()

        return torch.sqrt(poincare_dist+hypersphere_dist+euclidean_dist)

