import torch

def acosh(x):
    return torch.log(x + torch.sqrt(x**2-1))

# def dot(x,y): return torch.sum(x * y, -1)

# def dist_e(u, v):
#     """ Input shape (n, d) """
#     return torch.norm(u-v, 2, dim=1)

# def dist_s(u, v, eps=1e-9):
#     uu = SphericalParameter._proj(u)
#     vv = SphericalParameter._proj(v)
#     return torch.acos(torch.clamp(dot(uu, vv), -1+eps, 1-eps))


# def dist_p(u,v):
#     z  = 2*torch.norm(u-v,2,1)**2
#     uu = 1. + torch.div(z,((1-torch.norm(u,2,1)**2)*(1-torch.norm(v,2,1)**2)))
#     # machine_eps = np.finfo(uu.data.numpy().dtype).eps  # problem with cuda tensor
#     # return acosh(torch.clamp(uu, min=1+machine_eps))
#     return acosh(uu)

class ProductDistance():
    def __init__(self, x, y, k, atten_prob):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
        self.k = torch.tensor(k)
        self.atten_prob = atten_prob
        self.eps = 1e-8
    
    def dist_p(self):
        K = (1. / torch.sqrt(-self.k))
        z = 2*self.k*torch.norm(self.x-self.y,2,-1)**2
        dist = 1. - torch.div(z, ((1+self.k*torch.norm(self.x,2,-1)**2)*(1+self.k*torch.norm(self.y,2,-1)**2)))
        dist_ = acosh(torch.clamp(dist, min=1.0+self.eps))
        return K*dist_

    def dist_s(self):
        K = (1. / torch.sqrt(self.k))
        z = 2*self.k*torch.norm(self.x-self.y,2,-1)**2
        dist = 1. - torch.div(z, ((1+self.k*torch.norm(self.x,2,-1)**2)*(1+self.k*torch.norm(self.y,2,-1)**2)))
        dist_ = torch.acos(torch.clamp(dist, min=-1+self.eps, max=1-self.eps))
        return K*dist_

    def dist_e(self):
        return torch.norm(self.x-self.y, 2, -1)

    def productdistance(self):
        poincare_dist = self.dist_p()
        hypersphere_dist = self.dist_s()
        euclidean_dist = self.dist_e()

        return torch.sqrt(poincare_dist+hypersphere_dist+euclidean_dist)