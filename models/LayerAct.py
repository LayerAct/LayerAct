# importing 
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import warnings
warnings.filterwarnings('ignore')

def calculate_mean_std_for_forward(inputs, std = True) :     
    cal_dim = [1, 2, 3]
    mean = inputs.mean(dim=cal_dim, keepdim=True)
    if std : 
        std = inputs.std(dim=cal_dim, keepdim=True)
        return mean, std, cal_dim
    else : 
        return mean, cal_dim

def calculate_means_for_backward(t_1, t_2, cal_dim) : 
    t_1_mean = t_1.mean(dim=cal_dim, keepdim=True)
    t_2_mean = t_2.mean(dim=cal_dim, keepdim=True)
    
    return t_1_mean, t_2_mean

#############################################################

class LA_SiLU(nn.Module) : 
    def __init__(self, input_shape=None, affine=False, epsilon=1e-5) : 
        super(LA_SiLU, self).__init__()
        self.affine = affine
        self.epsilon = epsilon
        if self.affine : 
            self.gain = Parameter(torch.ones([1, input_shape]))
            self.bias = Parameter(torch.zeros([1, input_shape]))

    def forward(self, inputs) : 
        if self.affine : 
            return la_silu_affine.apply(inputs, self.gain, self.bias, self.epsilon)
        else : 
            return la_silu.apply(inputs, self.epsilon)


class la_silu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, epsilon):

        mean, std, cal_dim = calculate_mean_std_for_forward(inputs)
        nor = torch.div(torch.sub(inputs, mean), std+epsilon)
        scaler = torch.sigmoid(nor)
        z = torch.mul(scaler, inputs)

        ctx.save_for_backward(inputs, std, nor, scaler)
        ctx.cal_dim = cal_dim
        return z

    @staticmethod
    def backward(ctx, output_grad):
        inputs, std, nor, scaler = ctx.saved_tensors
        cal_dim = ctx.cal_dim
        
        inputs_grad = torch.mul(scaler, output_grad.clone())
        ds = torch.mul(inputs, output_grad.clone())
        dn = torch.mul(torch.mul(scaler, 1-scaler), ds)
        dn = torch.div(dn, std)
        dn_ = torch.mul(dn, nor)
        dn_1_mean, dn_2_mean = calculate_means_for_backward(dn, dn_, cal_dim)
        dn_2_mean = torch.mul(nor, dn_2_mean)
        dn = torch.sub(dn, torch.add(dn_1_mean, dn_2_mean))

        inputs_grad = torch.add(inputs_grad, dn)

        return inputs_grad, None, None


class la_silu_affine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, gain, bias, epsilon):
        shape = inputs.shape
        gain_dim = [0]
        if len(shape) > 2 : 
            gain = gain.reshape([s for s in gain.shape] + [1 for s in range(0, len(shape)-len(gain.shape))])
            bias = bias.reshape([s for s in bias.shape] + [1 for s in range(0, len(shape)-len(bias.shape))])
            gain_dim = [0, 2, 3]

        mean, std, cal_dim = calculate_mean_std_for_forward(inputs)
        nor = torch.div(torch.sub(inputs, mean), std+epsilon)
        aff = torch.add(torch.mul(nor, gain), bias)
        scaler = torch.sigmoid(aff)
        z = torch.mul(scaler, inputs)

        ctx.save_for_backward(inputs, gain, std, nor, scaler)
        ctx.cal_dim = cal_dim
        ctx.shape = shape
        ctx.gain_dim = gain_dim
        return z

    @staticmethod
    def backward(ctx, output_grad):
        inputs, gain, std, nor, scaler = ctx.saved_tensors
        cal_dim = ctx.cal_dim
        shape = ctx.shape 
        gain_dim = ctx.gain_dim
        inputs_grad = torch.mul(scaler, output_grad.clone()) 
        ds = torch.mul(inputs, output_grad.clone()) 
        dn = torch.mul(torch.mul(scaler, 1-scaler), ds) 

        gain_grad = torch.sum(torch.mul(dn, nor), dim=gain_dim, keepdim=True).reshape((1, shape[1]))
        bias_grad = torch.sum(dn, dim=gain_dim, keepdim=True).reshape((1, shape[1])) 

        dn = torch.mul(dn, gain) 
        dn = torch.div(dn, std) 
        dn_ = torch.mul(dn, nor) 
        dn_1_mean, dn_2_mean = calculate_means_for_backward(dn, dn_, cal_dim) 
        dn_2_mean = torch.mul(nor, dn_2_mean) 
        dn = torch.sub(dn, torch.add(dn_1_mean, dn_2_mean)) 

        inputs_grad = torch.add(inputs_grad, dn) 

        return inputs_grad, gain_grad, bias_grad, None, None
    

#############################################################

class LA_HardSiLU(nn.Module) : 
    def __init__(self, input_shape=None, a=3, affine=False, epsilon=1e-5) : 
        super(LA_HardSiLU, self).__init__()
        self.a = a
        self.affine = affine
        self.epsilon = epsilon
        if self.affine : 
            self.gain = Parameter(torch.ones([1, input_shape]))
            self.bias = Parameter(torch.zeros([1, input_shape]))

    def forward(self, inputs) : 
        if self.affine : 
            return la_hardsilu_affine.apply(inputs, self.a, self.gain, self.bias, self.epsilon)
        else : 
            return la_hardsilu.apply(inputs, self.a, self.epsilon)


class la_hardsilu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, a, epsilon):
        shape = inputs.shape
        device = inputs.device
        
        ones = torch.ones(shape, device=device)
        zeros = torch.zeros(shape, device=device)

        mean, std, cal_dim = calculate_mean_std_for_forward(inputs)
        nor = torch.div(torch.sub(inputs, mean), std+epsilon)
        scaler = torch.where(nor <= -1*a, zeros.clone(), nor.clone()/(a*2)+0.5)
        scaler = torch.where(nor >= a, ones.clone(), scaler)
        z = torch.mul(inputs, scaler)

        ctx.save_for_backward(inputs, std, nor, scaler, zeros)
        ctx.a = a
        ctx.cal_dim = cal_dim

        return z

    @staticmethod
    def backward(ctx, output_grad):
        inputs, std, nor, scaler, zeros = ctx.saved_tensors
        a = ctx.a
        cal_dim = ctx.cal_dim
        
        inputs_grad = torch.mul(output_grad.clone(), scaler)
        
        ds = torch.mul(output_grad.clone(), inputs)
        dn = torch.where(nor <= -a, zeros.clone(), ds/(a*2)) 
        dn = torch.where(nor >= a, zeros.clone(), dn) 
        dn = torch.div(dn, std) 
        dn_ = torch.mul(dn, nor) 
        dn_1_mean, dn_2_mean = calculate_means_for_backward(dn, dn_, cal_dim) 
        dn_2_mean = torch.mul(nor, dn_2_mean) 
        dn = torch.sub(dn, torch.add(dn_1_mean, dn_2_mean))

        inputs_grad = torch.add(inputs_grad, dn)
        
        return inputs_grad, None, None, None


class la_hardsilu_affine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, a, gain, bias, norm_type, epsilon):
        shape = inputs.shape
        device = inputs.device
        gain_dim = [0]
        if len(shape) > 2 : 
            gain = gain.reshape([s for s in gain.shape] + [1 for s in range(0, len(shape)-len(gain.shape))])
            bias = bias.reshape([s for s in bias.shape] + [1 for s in range(0, len(shape)-len(bias.shape))])
            gain_dim = [0, 2, 3]
        
        ones = torch.ones(shape, device=device)
        zeros = torch.zeros(shape, device=device)

        mean, std, cal_dim = calculate_mean_std_for_forward(inputs)
        nor = torch.div(torch.sub(inputs, mean), std+epsilon)
        aff = torch.add(torch.mul(nor, gain), bias)
        scaler = torch.where(aff <= -1*a, zeros.clone(), aff.clone()/(a*2)+0.5)
        scaler = torch.where(aff >= a, ones.clone(), scaler)
        z = torch.mul(inputs, scaler)

        ctx.save_for_backward(inputs, gain, std, nor, scaler, zeros)
        ctx.a = a
        ctx.cal_dim = cal_dim
        ctx.shape = shape 
        ctx.gain_dim = gain_dim

        return z

    @staticmethod
    def backward(ctx, output_grad):
        inputs, gain, std, nor, scaler, zeros = ctx.saved_tensors
        a = ctx.a
        cal_dim = ctx.cal_dim
        gain_dim = ctx.gain_dim
        shape = ctx.shape 
        
        inputs_grad = torch.mul(output_grad.clone(), scaler)
        
        ds = torch.mul(output_grad.clone(), inputs)
        dn = torch.where(nor <= -a, zeros.clone(), ds/(a*2)) 
        dn = torch.where(nor >= a, zeros.clone(), dn) 

        gain_grad = torch.sum(torch.mul(dn, nor), dim=gain_dim, keepdim=True).reshape((1, shape[1]))
        bias_grad = torch.sum(dn, dim=gain_dim, keepdim=True).reshape((1, shape[1])) 

        dn = torch.mul(dn, gain)
        dn = torch.div(dn, std) 
        dn_ = torch.mul(dn, nor) 
        dn_1_mean, dn_2_mean = calculate_means_for_backward(dn, dn_, cal_dim) 
        dn_2_mean = torch.mul(nor, dn_2_mean) 
        dn = torch.sub(dn, torch.add(dn_1_mean, dn_2_mean))

        inputs_grad = torch.add(inputs_grad, dn)
        
        return inputs_grad, None, gain_grad, bias_grad, None, None
    

#############################################################




