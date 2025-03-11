    
def dot(x,y): return torch.sum(x * y, -1)
def acosh(x):
    return torch.log(x + torch.sqrt(x**2-1))
# def tanh(x, clamp=15):
#     return x.clamp(-clamp, clamp).tanh()
# def tan(x, clamp=15):
#     return x.clamp(-clamp, clamp).tan()
def tanh(x, clamp=15):
    return torch.clamp(torch.tanh(x), -clamp, clamp)

def tan(x, clamp=15):
    return torch.clamp(torch.tan(x), -clamp, clamp)

def acos(x): 
    return torch.acos(x)

class ProductDistance():
    def __init__(self, x, y, k):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
        self.k = torch.tensor(k)
        if self.k < 0:
            self.k_ = torch.sqrt(-self.k)
        else:
            self.k_ = torch.sqrt(self.k)
        #self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
    
    @staticmethod
    def poincare_correct(x, eps=1e-10):
        current_norms = torch.norm(x,2,x.dim() - 1)
        mask_idx      = current_norms < 1./(1+eps)
        modified      = 1./((1+eps)*current_norms)
        modified[mask_idx] = 1.0
        #new_size      = [1]*current_norms.dim() + [x.size(x.dim()-1)]
        #return modified.unsqueeze(modified.dim()).repeat(*new_size)
        # return modified.unsqueeze(modified.dim()).expand(x.size())
        return modified.unsqueeze(-1)

    @staticmethod
    def poincare_proj(x, k, eps=1e-10):
        # if k<0:
        #     k = torch.sqrt(-k)
        # else:
        #     k = torch.sqrt(k)
        x = x*ProductDistance.poincare_correct(x)
        #print(f"poincare_correct:{x}")
        z = tanh(k*torch.norm(x, 2, -1))
        #print(f"p_exp_tanh:{z}")
        exp = torch.div(x*z, (k*torch.norm(x, 2, -1)))
        #print(f"p_exp:{exp}")
        #print(f"poincare_exp: {exp}")
        return exp
        
    @staticmethod
    def hypersphere_proj(x, k):
        z = tan(k*torch.norm(x, 2, -1))
        #print(f"h_exp_tan:{z}")
        exp = torch.div(x*z, (k*torch.norm(x, 2, -1)))
        #print(f"h_exp: {exp}")
        return exp

    def hypersphere_dist(self, eps=1e-7):
        proj_x = ProductDistance.hypersphere_proj(self.x, self.k)
        proj_y = ProductDistance.hypersphere_proj(self.y, self.k)
        K = 1/self.k_
        #print(K)
        z  = 2*self.k*((torch.norm(proj_x-proj_y,2,-1))**2)
        #print(f"h_dist_z:{z}")
        uu =  1. + torch.div(z,((1+self.k*((torch.norm(proj_x,2,-1))**2))*(1+self.k*((torch.norm(proj_y,2,-1))**2))))
        #print(f"hypersphere_uu_for_clamp: {uu}")
        #return K*torch.acos(torch.clamp(uu), -1+eps, 1-eps
        valid_mask = (uu >= -1) & (uu <= 1)
        uu_valid = uu[valid_mask]
        uu_ = torch.empty_like(uu)
        uu_[valid_mask] = acos(uu_valid)  
        #h_ = acos(uu)
        #print(f"h_:{h_}")
        #clipped = torch.max(torch.min(x, max), min)
        min_, max_ = torch.tensor(-1+eps), torch.tensor(1-eps)
        h_dist = K*(torch.max(torch.min(uu_, max_), min_))
        #print(f"h_uu:{uu_}")
        #print(f"h_dist:{h_dist}")
        return h_dist

    def poincare_dist(self):
        proj_x = ProductDistance.poincare_proj(self.x, self.k)
        proj_y = ProductDistance.poincare_proj(self.y, self.k)
        K = 1/self.k_
        #print(K)
        z  = 2*self.k*((torch.norm(proj_x-proj_y,2,-1))**2)
        #print(f"p_dist_z:{z}")
        uu =  1. + torch.div(z,((1+self.k*(torch.norm(proj_x,2,-1)**2))*(1+self.k*(torch.norm(proj_y,2,-1)**2))))
        #print(f"p_uu:{uu}")
        #print(f"p_dist:{K*acosh(uu)}")
        return K*acosh(uu)

    def hypersphere_exp(x, k_):
        #self.x[..., 0] = torch.sqrt(1 - torch.norm(self.x[..., 1:],2,-1)**2)
        x[...,0] = k_
        n = torch.norm(x, 2, -1, keepdim=True)
        mask = torch.abs(n)<1e-7
        cos = torch.cos(n)
        cos[mask] = 1.0
        sin = torch.sin(n)
        sin[mask] = 0.0
        n[torch.abs(n)<1e-7] = 1.0
        e = cos*x + sin*x/n
        return e/torch.norm(e, 2, -1, True)


    def hypersphere_mfd(self, eps=1e-9): #to calculate the hypersphere distance on full mfd
        proj_x = ProductDistance.hypersphere_exp(self.x, self.k_)
        proj_y = ProductDistance.hypersphere_exp(self.y, self.k_)
        K = 1/self.k_
        dist_ = self.k_*torch.clamp(dot(proj_x, proj_y), -1+eps, 1-eps)
        return K*torch.acos(dist_)




    def euclidean_dist(self):
        """ Input shape (n, d) """
        #print(f"e_dist:{torch.norm(self.x-self.y, 2, dim=-1)}")
        return torch.norm(self.x-self.y, 2, dim=-1)
