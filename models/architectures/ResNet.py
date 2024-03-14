# We customized the implementation from PyTorch. 
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
import numpy as np
import random
import os
import time
import copy
import math

import torch
import torch.nn as nn
from torch import Tensor

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

def random_seed_set(rs) : 
    torch.manual_seed(rs)
    torch.cuda.manual_seed(rs)
    torch.cuda.manual_seed_all(rs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    torch.backends.cudnn.enabled = False
    np.random.seed(rs)
    random.seed(rs)
    os.environ["PYTHONHASHSEED"] = str(rs)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        activation, 
        activation_params, 
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.act1 = activation(**activation_params)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.act2 = activation(**activation_params)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act2(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        H, 
        W,
        activation, 
        activation_params, 
        normalization, 
        normalization_params, 
        inplanes: int,
        planes: int,
        stride: int = 1,
        do_downsample: Optional[nn.Module] = None,
        down_sample_dict = {},
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        self.H, self.W = H, W
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = normalization_set(normalization, normalization_params, width, self.H, self.W, in_block=True)
        self.bn1 = norm_layer(width)
        self.act1 = activation(**activation_params)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)

        self.H = math.floor((self.H+2*1-dilation*(3-1)-1)/stride+1)
        self.W = math.floor((self.W+2*1-dilation*(3-1)-1)/stride+1)

        self.bn2 = normalization_set(normalization, normalization_params, width, self.H, self.W, in_block=True)
        self.act2 = activation(**activation_params)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = normalization_set(normalization, normalization_params, planes * self.expansion, self.H, self.W, in_block=True)
        self.act3 = activation(**activation_params)
        
        if do_downsample : 
            downsample = nn.Sequential(
                conv1x1(down_sample_dict['in'], down_sample_dict['out'], down_sample_dict['stride']),
                normalization_set(normalization, normalization_params, down_sample_dict['out'], self.H, self.W, in_block=True)
            )
        else : 
            downsample=None

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        activation, 
        activation_params,
        normalization, 
        normalization_params, 
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        #norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.normalization = normalization
        self.normalization_params = normalization_params

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        
        H = 224
        W = 224

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        H = ((H+2*3-1*(7-1)-1)/2+1)
        W = ((W+2*3-1*(7-1)-1)/2+1)

        self.bn1 = normalization_set(normalization, normalization_params, self.inplanes, H, W)
        self.act1 = activation(**activation_params)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1, H, W = self._make_layer(H, W, activation, activation_params, block, 64, layers[0])
        self.layer2, H, W = self._make_layer(H, W, activation, activation_params, block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3, H, W = self._make_layer(H, W, activation, activation_params, block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4, H, W = self._make_layer(H, W, activation, activation_params, block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        H, 
        W,
        activation, 
        activation_params,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        do_downsample = False
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            do_downsample = True
            down_sample_dict = {'in' : self.inplanes, 'out' : planes * block.expansion, 'stride' : stride}

        layers = []
        for i in range(0, blocks) : 
            if i == 0 : 
                current =  block(
                    H, W, activation, activation_params, self.normalization, self.normalization_params,
                    self.inplanes, planes, stride, do_downsample, down_sample_dict,
                    self.groups, self.base_width, previous_dilation
                )
                self.inplanes = planes * block.expansion
            else : 
                current = block(
                    H, W,
                    activation, 
                    activation_params,
                    self.normalization, 
                    self.normalization_params,
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation
                )
            H, W = current.H, current.W
            layers.append(current)
        

        return nn.Sequential(*layers), H, W

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        #print('in' + ' '*50, end='\r')
        return self._forward_impl(x)
    
def _resnet(
    activation, 
    activation_params, 
    normalization, 
    normalization_params, 
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    groups: int = 1,
    width_per_group: int = 64,
    **kwags: Any, 
) -> ResNet:

    model = ResNet(activation, activation_params, normalization, normalization_params, block, layers, **kwags)

    return model
    
def resnet18(activation, activation_params, normalization, normalization_params, num_classes) :
    return _resnet(activation, activation_params, normalization, normalization_params, BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    
def resnet32(activation, activation_params, normalization, normalization_params, num_classes) :
    return _resnet(activation, activation_params, normalization, normalization_params, BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    
def resnet50(activation, activation_params, normalization, normalization_params, num_classes) :
    return _resnet(activation, activation_params, normalization, normalization_params, Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

def resnet101(activation, activation_params, normalization, normalization_params, num_classes) :
    return _resnet(activation, activation_params, normalization, normalization_params, Bottleneck, [3, 4, 23, 3], num_classes=num_classes)

def resnet152(activation, activation_params, normalization, normalization_params, num_classes) :
    return _resnet(activation, activation_params, normalization, normalization_params, Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

def resnext50_32x4d(activation, activation_params, normalization, normalization_params, num_classes) : 
    return _resnet(activation, activation_params, normalization, normalization_params, Bottleneck, [3, 8, 36, 3], num_classes=num_classes, groups=32, width_per_group=4)

