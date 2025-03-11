    
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
        print(f"poincare_correct:{x}")
        z = tanh(k*torch.norm(x, 2, -1))
        print(f"p_exp_tanh:{z}")
        exp = torch.div(x*z, (k*torch.norm(x, 2, -1)))
        print(f"p_exp:{exp}")
        #print(f"poincare_exp: {exp}")
        return exp
        
    @staticmethod
    def hypersphere_proj(x, k):
        z = tan(k*torch.norm(x, 2, -1))
        print(f"h_exp_tan:{z}")
        exp = torch.div(x*z, (k*torch.norm(x, 2, -1)))
        print(f"h_exp: {exp}")
        return exp

    def hypersphere_dist(self, eps=1e-9):
        proj_x = ProductDistance.hypersphere_proj(self.x, self.k)
        proj_y = ProductDistance.hypersphere_proj(self.y, self.k)
        K = 1/self.k_
        #print(K)
        z  = 2*self.k*((torch.norm(proj_x-proj_y,2,-1))**2)
        print(f"h_dist_z:{z}")
        uu =  1. + torch.div(z,((1+self.k*((torch.norm(proj_x,2,-1))**2))*(1+self.k*((torch.norm(proj_y,2,-1))**2))))
        #print(f"hypersphere_uu_for_clamp: {uu}")
        #return K*torch.acos(torch.clamp(uu), -1+eps, 1-eps
        print(f"h_uu:{uu}")
        print(f"h_dist:{K*torch.acos(uu)}")
        return K*(torch.acos(uu))

    def poincare_dist(self):
        proj_x = ProductDistance.poincare_proj(self.x, self.k)
        proj_y = ProductDistance.poincare_proj(self.y, self.k)
        K = 1/self.k_
        #print(K)
        z  = 2*self.k*((torch.norm(proj_x-proj_y,2,-1))**2)
        print(f"p_dist_z:{z}")
        uu =  1. + torch.div(z,((1+self.k*(torch.norm(proj_x,2,-1)**2))*(1+self.k*(torch.norm(proj_y,2,-1)**2))))
        print(f"p_uu:{uu}")
        print(f"p_dist:{K*acosh(uu)}")
        return K*acosh(uu)

    def euclidean_dist(self):
        """ Input shape (n, d) """
        return torch.norm(self.x-self.y, 2, dim=-1)
