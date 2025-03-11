def dot(x,y): return torch.sum(x * y, -1)
def acosh(x):
    return torch.log(x + torch.sqrt(x**2-1))

class RParameter(nn.Parameter):
    def __new__(cls, data=None, requires_grad=True, sizes=None, exp=False):
        if data is None:
            assert sizes is not None
            data = (1e-3 * torch.randn(sizes, dtype=torch.double)).clamp_(min=-3e-3,max=3e-3)
        #TODO get partial data if too big i.e. data[0:n,0:d]
        ret =  super().__new__(cls, data, requires_grad=requires_grad)
        # ret.data    = data
        ret.initial_proj()
        ret.use_exp = exp
        return ret

    @staticmethod
    def _proj(x):
        raise NotImplemented

    def proj(self):
        self.data = self.__class__._proj(self.data.detach())
        # print(torch.norm(self.data, dim=-1))

    def initial_proj(self):
        """ Project the initialization of the embedding onto the manifold """
        self.proj()

    def modify_grad_inplace(self):
        pass

    @staticmethod
    def correct_metric(ps):
        for p in ps:
            if isinstance(p,RParameter):
                p.modify_grad_inplace()

class PoincareParameter(RParameter):
    def __new__(cls, data=None, requires_grad=True, sizes=None, check_graph=False):
        ret =  super().__new__(cls, data, requires_grad, sizes)
        ret.check_graph = check_graph
        return ret

    def modify_grad_inplace(self):
        # d        = self.data.dim()
        w_norm   = torch.norm(self.data,2,-1, True)
        # This is the inverse of the remanian metric, which we need to correct for.
        hyper_b  = (1 - w_norm**2)**2/4
        # new_size = tuple([1] * (d - 1) + [self.data.size(d-1)])
        # self.grad   *= hyper_b.repeat(*new_size) # multiply pointwise
        self.grad   *= hyper_b # multiply pointwise
        self.grad.clamp_(min=-10000.0, max=10000.0)

        # We could do the projection here?
        # NB: THIS IS DEATHLY SLOW. FIX IT
        if self.check_graph and np.any(np.isnan(self.grad.data.cpu().numpy())):
             print(np.any(np.isnan(self.data.cpu().numpy())))
             print(np.any(np.isnan(self.grad.data.cpu().numpy())))
             print(np.any(np.isnan(w_norm.cpu().numpy())))
             raise ValueError("NaN During Hyperbolic")

    @staticmethod
    def _correct(x, eps=1e-10):
        current_norms = torch.norm(x,2,x.dim() - 1)
        mask_idx      = current_norms < 1./(1+eps)
        modified      = 1./((1+eps)*current_norms)
        modified[mask_idx] = 1.0
        #new_size      = [1]*current_norms.dim() + [x.size(x.dim()-1)]
        #return modified.unsqueeze(modified.dim()).repeat(*new_size)
        # return modified.unsqueeze(modified.dim()).expand(x.size())
        return modified.unsqueeze(-1)

    @staticmethod
    def _proj(x, eps=1e-10):
        return x * PoincareParameter._correct(x, eps=eps)

    # def proj(self, eps=1e-10):
    #     self.data = self.__class__._proj(self.data.detach())#PoincareParameter._correct(self.data, eps=eps)

    def __repr__(self):
        return 'Hyperbolic parameter containing:' + self.data.__repr__()

class SphericalParameter(RParameter):
    def __new__(cls, data=None, requires_grad=True, sizes=None, exp=True):
        if sizes is not None:
            sizes = list(sizes)
            sizes[-1] += 1
        return super().__new__(cls, data, requires_grad, sizes, exp)


    def modify_grad_inplace(self):
        """ Convert Euclidean gradient into Riemannian by projecting onto tangent space """
        # pass
        self.grad -= dot(self.data, self.grad).unsqueeze(-1) * self.data

    def exp(self, lr):
        x = self.data.detach()
        v = -lr*self.grad

        retract = False
        if retract:
        # retraction
            self.data = x + v

        else:
            n = torch.norm(v, 2, -1, keepdim=True)
            mask = torch.abs(n)<1e-7
            cos = torch.cos(n)
            cos[mask] = 1.0
            sin = torch.sin(n)
            sin[mask] = 0.0
            n[torch.abs(n)<1e-7] = 1.0
            e = cos*x + sin*v/n
            self.data = e
        self.proj()

    @staticmethod
    def _proj(x):
        # return x / torch.norm(x, 2, -1).unsqueeze(-1)
        return x / torch.norm(x, 2, -1, True) 


    # def proj(self):
    #     x = self.data.detach()
    #     self.data = SphericalParameter._proj(x)
    def initial_proj(self):
        # pass
        self.data[...,0] = torch.sqrt(1 - torch.norm(self.data[...,1:],2,-1)**2)

class EuclideanParameter(RParameter):
    def proj(x):
        pass

def dist_p(u,v, k):
    k = torch.tensor(k)
    if k <0:
      k = 1./torch.sqrt(-k)
    else:
      k = 1./torch.sqrt(k)
    z  = 2*k*(torch.norm(u-v,2,-1)**2)
    print(k)
    #uu = 1. + torch.div(z,((1-torch.norm(u,2,-1)**2)*(1-torch.norm(v,2,-1)**2))) K=1
    uu =  1. + torch.div(z,((1+k*(torch.norm(u,2,-1)**2))*(1+k*(torch.norm(v,2,-1)**2)))) #K=k
    print(uu)
    # machine_eps = np.finfo(uu.data.numpy().dtype).eps  # problem with cuda tensor
    # return acosh(torch.clamp(uu, min=1+machine_eps))
    return k*acosh(uu)

def dot(x,y): return torch.sum(x * y, -1)

def dist_e(u, v):
    """ Input shape (n, d) """
    return torch.norm(u-v, 2, dim=-1)

def dist_s(u, v, k, eps=1e-9):
    uu = SphericalParameter._proj(u)
    vv = SphericalParameter._proj(v)
    print(k)
    #K=k
    k = torch.tensor(k)
    if k <0:
      k = 1./torch.sqrt(-k)
    else:
      k = 1./torch.sqrt(k)
    z = torch.clamp(k*dot(uu, vv), -1+eps, 1-eps)
    return k*torch.acos(z)