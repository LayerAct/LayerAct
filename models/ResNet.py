import torch
import torch.nn as nn
import math
import os
import random
import numpy as np
import torch.utils.model_zoo as model_zoo
import time
    
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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, activation, activation_params, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        try : 
            self.act1 = activation(**activation_params)
        except :
            self.act1 = activation(planes, **activation_params)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        try : 
            self.act2 = activation(**activation_params)
        except :
            self.act2 = activation(planes, **activation_params)
        
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None : 
            identity = self.downsample(x)

        out += identity
        out = self.act2(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self, 
        activation, activation_params, block, layers, init_name='kaiming', num_classes=10, rs=0
        ):
        self.inplanes = 16
        super(ResNet, self).__init__()

        random_seed_set(rs)
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        try : 
            self.act = activation(**activation_params)
        except :
            self.act = activation(16, **activation_params)
        self.layer1 = self._make_layer(activation, activation_params, block, 16, layers[0])
        self.layer2 = self._make_layer(activation, activation_params, block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(activation, activation_params, block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)
        
        self.set_init(init_name)

    def _make_layer(self, activation, activation_params, block, planes, blocks, stride=1):
        
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion : 
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes*block.expansion, stride), 
                nn.BatchNorm2d(planes*block.expansion)
            )
        layers = []
        layers.append(block(activation, activation_params, self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(activation, activation_params, self.inplanes, planes))

        return nn.Sequential(*layers)

    def set_init(self, init_name) : 
        if init_name == 'kaiming' : 
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) :
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d) :
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        elif init_name == 'xavier' :  
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) :
                    nn.init.xavier_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d) :
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




def resnet20(activation, activation_params={}, init_name='kaiming', rs=0, num_classes=10):
    return ResNet(activation, activation_params, BasicBlock, [3, 3, 3], init_name=init_name, num_classes=num_classes, rs=rs)

def resnet32(activation, activation_params={}, init_name='kaiming', rs=0, num_classes=10):
    return ResNet(activation, activation_params, BasicBlock, [5, 5, 5], init_name=init_name, num_classes=num_classes, rs=rs)

def resnet44(activation, activation_params={}, init_name='kaiming', rs=0, num_classes=10):
    return ResNet(activation, activation_params, BasicBlock, [7, 7, 7], init_name=init_name, num_classes=num_classes, rs=rs)

def resnet56(activation, activation_params={}, init_name='kaiming', rs=0, num_classes=10):
    return ResNet(activation, activation_params, BasicBlock, [9, 9, 9], init_name=init_name, num_classes=num_classes, rs=rs)

def resnet110(activation, activation_params={}, init_name='kaiming', rs=0, num_classes=10):
    return ResNet(activation, activation_params, BasicBlock, [18, 18, 18], init_name=init_name, num_classes=num_classes, rs=rs)

