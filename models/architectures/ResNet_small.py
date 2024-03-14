# We customized the implementation from PyTorch. 
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

import torch.nn as nn
import torch.nn.functional as F

import math
import copy 

from ..modules.instance_enhancement_batch_normalization import BAN2d
from ..modules.switchable_normalization import SwitchNorm2d
from ..modules.decorrelated_batch_normalization import DBN, DBN2, ZCANormBatch

def normalization_set(normalization, normalization_params, channel_size, H=None, W=None, last=False, in_block=False) :
    if isinstance(normalization, list) : 
        current_normalization = []
        for n in normalization : 
            if n == nn.BatchNorm2d : 
                current_normalization.append(n(channel_size, **normalization_params))
            elif n == BAN2d : 
                current_normalization.append(n(channel_size, **normalization_params))
            elif n == SwitchNorm2d : 
                if last : current_normalization.append(n(channel_size, using_moving_average=True, using_bn=True, last_gamma=True))
                else : current_normalization.append(n(channel_size, using_moving_average=True, using_bn=True))
            elif n == DBN : 
                if 'num_groups' not in normalization_params.keys() : 
                    if channel_size % 32 != 0 : normalization_params['num_groups'] = channel_size/2
                else : 
                    if channel_size % normalization_params['num_groups'] != 0 : normalization_params['num_groups'] = int(channel_size)
                current_normalization.append(n(channel_size, **normalization_params))
            elif n == DBN2 : 
                if 'num_groups' not in normalization_params.keys() : 
                    if channel_size % 32 != 0 : normalization_params['num_groups'] = channel_size/2
                else : 
                    if channel_size % normalization_params['num_groups'] != 0 : normalization_params['num_groups'] = int(channel_size)
                current_normalization.append(n(channel_size, **normalization_params))
            else : 
                current_normalization.append(n([channel_size, H, W], **normalization_params))
        return nn.Sequential(*current_normalization)
    else : 
        if normalization == nn.BatchNorm2d : 
            return normalization(channel_size, **normalization_params)
        elif normalization == BAN2d : 
            return normalization(channel_size, **normalization_params)
        elif normalization == SwitchNorm2d : 
            if last : return normalization(channel_size, using_moving_average=True, using_bn=True, last_gamma=True)
            else : return normalization(channel_size, using_moving_average=True, using_bn=True)
        elif normalization == DBN : 
            if in_block : 
                return nn.BatchNorm2d(channel_size)
            else : 
                norm_params = copy.deepcopy(normalization_params)
                if 'num_groups' not in norm_params.keys() : 
                    if channel_size % 32 != 0 : norm_params['num_groups'] = int(channel_size)
                else : 
                    if channel_size % norm_params['num_groups'] != 0 : norm_params['num_groups'] = int(channel_size)
                return normalization(channel_size, **norm_params)
        elif normalization == DBN2 : 
            if in_block : 
                return nn.BatchNorm2d(channel_size)
            else : 
                norm_params = copy.deepcopy(normalization_params)
                if 'num_groups' not in norm_params.keys() : 
                    if channel_size % 32 != 0 : norm_params['num_groups'] = int(channel_size)
                else : 
                    if channel_size % norm_params['num_groups'] != 0 : norm_params['num_groups'] = int(channel_size)
                return normalization(channel_size, **norm_params)
        elif normalization == ZCANormBatch : 
            if in_block : 
                return nn.BatchNorm2d(channel_size)
            else : 
                norm_params = copy.deepcopy(normalization_params)
                if 'num_groups' not in norm_params.keys() : 
                    if channel_size % 32 != 0 : norm_params['num_groups'] = int(channel_size)
                else : 
                    if channel_size % norm_params['num_groups'] != 0 : norm_params['num_groups'] = int(channel_size)
                return normalization(channel_size, **norm_params)
        else : 
            return normalization([channel_size, H, W], **normalization_params)



class ResNet(nn.Module):
    
    def __init__(self, activation, activation_params, normalization, normalization_params, layers, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        H = math.floor((32 + 2 - 2 - 1)+1)
        W = math.floor((32 + 2 - 2 - 1)+1)
        self.norm1 = normalization_set(normalization, normalization_params, 16, H, W)
        self.act1 = activation(**activation_params)
        self.layers1, H, W = self._make_layer( activation, activation_params, normalization, normalization_params, layers[0], 16, 16, 1, H, W)
        self.layers2, H, W = self._make_layer( activation, activation_params, normalization, normalization_params, layers[1], 32, 16, 2, H, W)
        self.layers3, H, W = self._make_layer( activation, activation_params, normalization, normalization_params, layers[2], 64, 32, 2, H, W)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(64, num_classes)
    
    def _make_layer(self, activation, activation_params, normalization, normalization_params, layer_count, channels, channels_in, stride, H, W):
        layers = []
        for _ in range(0, layer_count) : 
            if _ == 0 : 
                current = ResBlock(H, W, activation, activation_params, normalization, normalization_params, channels, channels_in, stride)
            else : 
                current = ResBlock(H, W, activation, activation_params, normalization, normalization_params, channels)
            layers.append(current)
            H, W = current.H, current.W

        return nn.Sequential(*layers), H, W
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResBlock(nn.Module):
    
    def __init__(self, H, W, activation, activation_params, normalization, normalization_params, num_filters, channels_in=None, stride=1):
        super(ResBlock, self).__init__()
        
        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == num_filters:
            channels_in = num_filters
            self.projection = None
        else : 
            self.projection = IdentityPadding(num_filters, channels_in, stride)

        self.conv1 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        self.H = math.floor((H + 2 - 2 - 1)/stride + 1)
        self.W = math.floor((W + 2 - 2 - 1)/stride + 1)

        self.bn1 = normalization_set(normalization, normalization_params, num_filters, self.H, self.W, in_block=True)
        self.act1 = activation(**activation_params)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.H = math.floor((self.H + 2 - 2 - 1) + 1)
        self.W = math.floor((self.W + 2 - 2 - 1) + 1)

        self.bn2 = normalization_set(normalization, normalization_params, num_filters, self.H, self.W, last=True, in_block=True)
        self.act2 = activation(**activation_params)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.projection:
            residual = self.projection(x)

        out += residual
        out = self.act2(out)
        return out


# various projection options to change number of filters in residual connection
# option A from paper
class IdentityPadding(nn.Module):
    def __init__(self, num_filters, channels_in, stride):
        super(IdentityPadding, self).__init__()
        # with kernel_size=1, max pooling is equivalent to identity mapping with stride
        self.identity = nn.MaxPool2d(1, stride=stride)
        self.num_zeros = num_filters - channels_in
    
    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out
    
def resnet20( activation, activation_params, normalization, normalization_params, num_classes) :
    return ResNet( activation, activation_params, normalization, normalization_params, [3, 3, 3], num_classes=num_classes)
    
def resnet32( activation, activation_params, normalization, normalization_params, num_classes) :
    return ResNet( activation, activation_params, normalization, normalization_params, [5, 5, 5], num_classes=num_classes)
    
def resnet44( activation, activation_params, normalization, normalization_params, num_classes) :
    return ResNet( activation, activation_params, normalization, normalization_params, [7, 7, 7], num_classes=num_classes)


