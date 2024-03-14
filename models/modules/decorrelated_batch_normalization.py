# We used the impementation from Huang et al., "Iterative Normalization: Beyond Standardization towards Efficient Whitening", CVPR, 2019. 
# https://github.com/huangleiBuaa/IterNorm-pytorch

import torch
import torch.nn as nn
from torch.nn import Parameter

from mpmath import *
import numpy as np


class DBN(nn.Module):
    def __init__(self, num_features, num_groups=16, num_channels=0, dim=4, eps=1e-5, momentum=0.1, affine=True, mode=0,
                 *args, **kwargs):
        super(DBN, self).__init__()
        #print(num_features, num_groups)
        if num_channels > 0:
            num_groups = num_features // num_channels
        self.num_features = num_features
        self.num_groups = num_groups
        assert self.num_features % self.num_groups == 0
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.mode = mode

        self.shape = [1] * dim
        self.shape[1] = num_features

        if self.affine:
            self.weight = Parameter(torch.Tensor(*self.shape))
            self.bias = Parameter(torch.Tensor(*self.shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_groups, 1))
        self.register_buffer('running_projection', torch.eye(num_groups))
        self.reset_parameters()

    # def reset_running_stats(self):
    #     self.running_mean.zero_()
    #     self.running_var.eye_(1)

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor):
        size = input.size()
        assert input.dim() == self.dim and size[1] == self.num_features
        x = input.view(size[0] * size[1] // self.num_groups, self.num_groups, *size[2:])
        training = self.mode > 0 or (self.mode == 0 and self.training)
        x = x.transpose(0, 1).contiguous().view(self.num_groups, -1)
        if training:
            mean = x.mean(1, keepdim=True)
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean
            x_mean = x - mean
            sigma = x_mean.matmul(x_mean.t()) / x.size(1) + self.eps * torch.eye(self.num_groups, device=input.device)
            # print('sigma size {}'.format(sigma.size()))
            u, eig, _ = sigma.svd()
            scale = eig.rsqrt()
            wm = u.matmul(scale.diag()).matmul(u.t())
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * wm
            y = wm.matmul(x_mean)
        else:
            x_mean = x - self.running_mean
            y = self.running_projection.matmul(x_mean)
        output = y.view(self.num_groups, size[0] * size[1] // self.num_groups, *size[2:]).transpose(0, 1)
        output = output.contiguous().view_as(input)
        if self.affine:
            output = output * self.weight + self.bias
        return output

    def extra_repr(self):
        return '{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'mode={mode}'.format(**self.__dict__)


class DBN2(DBN):
    """
    when evaluation phase, sigma using running average.
    """

    def forward(self, input: torch.Tensor):
        size = input.size()
        assert input.dim() == self.dim and size[1] == self.num_features
        x = input.view(size[0] * size[1] // self.num_groups, self.num_groups, *size[2:])
        training = self.mode > 0 or (self.mode == 0 and self.training)
        x = x.transpose(0, 1).contiguous().view(self.num_groups, -1)
        mean = x.mean(1, keepdim=True) if training else self.running_mean
        x_mean = x - mean
        if training:
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean
            sigma = x_mean.matmul(x_mean.t()) / x.size(1) + self.eps * torch.eye(self.num_groups, device=input.device)
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * sigma
        else:
            sigma = self.running_projection
        u, eig, _ = sigma.svd()
        scale = eig.rsqrt()
        wm = u.matmul(scale.diag()).matmul(u.t())
        y = wm.matmul(x_mean)
        output = y.view(self.num_groups, size[0] * size[1] // self.num_groups, *size[2:]).transpose(0, 1)
        output = output.contiguous().view_as(input)
        if self.affine:
            output = output * self.weight + self.bias
        return output
    
mp.dps = 32
one = mpf(1)
mp.pretty = True

def f(x):
    return sqrt(one-x)

a = taylor(f, 0, 10)
pade_p, pade_q = pade(a, 5, 5)
a = torch.from_numpy(np.array(a).astype(float))
pade_p = torch.from_numpy(np.array(pade_p).astype(float))
pade_q = torch.from_numpy(np.array(pade_q).astype(float))

def matrix_pade_approximant(p,I):
    p_sqrt = pade_p[0]*I
    q_sqrt = pade_q[0]*I
    p_app = I - p
    p_hat = p_app
    for i in range(5):
        p_sqrt += pade_p[i+1]*p_hat
        q_sqrt += pade_q[i+1]*p_hat
        p_hat = p_hat.mm(p_app)
    #There are 4 options to compute the MPA: comput Matrix Inverse or Matrix Linear System on CPU/GPU;
    #It seems that single matrix is faster on CPU and batched matrices are faster on GPU
    #Please check which one is faster before running the code;
    return torch.linalg.solve(q_sqrt, p_sqrt)
    #return torch.linalg.solve(q_sqrt.cpu(), p_sqrt.cpu()).cuda()
    #return torch.linalg.inv(q_sqrt).mm(p_sqrt)
    #return torch.linalg.inv(q_sqrt.cpu()).cuda().bmm(p_sqrt)
    

class MPA_Lya(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M):
        normM = torch.norm(M)
        I = torch.eye(M.size(0), requires_grad=False, device=M.device)
        #M_sqrt = matrix_taylor_polynomial(M/normM,I)
        M_sqrt = matrix_pade_approximant(M / normM, I)
        M_sqrt = M_sqrt * torch.sqrt(normM)
        ctx.save_for_backward(M, M_sqrt, normM,  I)
        return M_sqrt

    @staticmethod
    def backward(ctx, grad_output):
        M, M_sqrt, normM,  I = ctx.saved_tensors
        b = M_sqrt / torch.sqrt(normM)
        c = grad_output / torch.sqrt(normM)
        for i in range(8):
            #In case you might terminate the iteration by checking convergence
            #if th.norm(b-I)<1e-4:
            #    break
            b_2 = b.mm(b)
            c = 0.5 * (c.mm(3.0*I-b_2)-b_2.mm(c)+b.mm(c).mm(b))
            b = 0.5 * b.mm(3.0 * I - b_2)
        grad_input = 0.5 * c
        return grad_input


class ZCANormBatch(nn.Module):
    def __init__(self, num_features, num_groups=1, eps=1e-2, momentum=0.1, affine=True):
        super(ZCANormBatch, self).__init__()
        print('ZCANormBatch Group Num {}'.format(num_groups))
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = num_groups
        self.weight = Parameter(torch.Tensor(num_groups, int(num_features/num_groups), 1))
        self.bias = Parameter(torch.Tensor(num_groups, int(num_features/num_groups), 1))
        #Matrix Square Root or Inverse Square Root layer
        self.svdlayer = MPA_Lya.apply
        self.register_buffer('running_mean', torch.zeros(num_groups, int(num_features/num_groups), 1))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        self.register_buffer("running_subspace", torch.eye(length, length).view(1,length,length).repeat(self.groups,1,1))

    def reset_running_stats(self):
            self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.training:
            N, C, H, W = x.size()
            G = self.groups
            n_mem = C // G
            x = x.transpose(0, 1).contiguous().view(G, n_mem, -1)
            mu = x.mean(2, keepdim=True)
            x = x - mu
            xxt = torch.bmm(x, x.transpose(1,2)) / (N * H * W) + torch.eye(n_mem, out=torch.empty_like(x)).unsqueeze(0) * self.eps
            assert C % G == 0
            subspace = torch.inverse(self.svdlayer(xxt))
            xr = torch.bmm(subspace, x)
            with torch.no_grad():
                running_subspace = self.__getattr__('running_subspace')
                running_subspace.data = (1 - self.momentum) * running_subspace.data + self.momentum * subspace.data
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            xr = xr * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)
            return xr

        else:
            N, C, H, W = x.size()
            G = self.groups
            n_mem = C // G
            x = x.transpose(0, 1).contiguous().view(G, n_mem, -1)
            x = (x - self.running_mean)
            subspace = self.__getattr__('running_subspace')
            x= torch.bmm(subspace, x)
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(ZCANormBatch, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


if __name__ == '__main__':
    dbn = DBN(64, 32, affine=False, momentum=1.)
    x = torch.randn(2, 64, 7, 7)
    print(dbn)
    y = dbn(x)
    print('y size:', y.size())
    y = y.view(y.size(0) * y.size(1) // dbn.num_groups, dbn.num_groups, *y.size()[2:])
    y = y.transpose(0, 1).contiguous().view(dbn.num_groups, -1)
    print('y reshaped:', y.size())
    z = y.matmul(y.t())
    print('train mode:', z.diag())
    dbn.eval()
    y = dbn(x)
    y = y.view(y.size(0) * y.size(1) // dbn.num_groups, dbn.num_groups, *y.size()[2:])
    y = y.transpose(0, 1).contiguous().view(dbn.num_groups, -1)
    z = y.matmul(y.t())
    print('eval mode:', z.diag())
    print(__file__)