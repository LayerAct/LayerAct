import torch.nn as nn 

from .modules.instance_enhancement_batch_normalization import BAN2d
from .modules.switchable_normalization import SwitchNorm2d
from .modules.decorrelated_batch_normalization import DBN, DBN2, ZCANormBatch
from .modules.layer_level_activation import LA_SiLU, LA_HardSiLU

from .architectures.ResNet import resnet18, resnet50, resnet101, resnet152, resnext50_32x4d
from .architectures.ResNet_small import resnet20, resnet32, resnet44
from .architectures.unet import UNet

__all__ = ['model_building']

def normaliation_building(args) : 
    normalization_params = {}
    n_list = args.model.split('_')
    if 'ln' in args.model or 'layernorm' in args.model : 
        if 'noaffine' in args.model : 
            normalization_params = {'elementwise_affine' : False}
        normalization = nn.LayerNorm
        normalization_params['eps'] = float(
            [n for n in n_list if 'eps' in n][0].split('eps')[-1]
            ) if 'eps' in args.model else 1e-5
        
    elif 'nonorm' in args.model or 'nn' in args.model : normalization = []
    elif 'iebn' in args.model : 
        normalization = BAN2d
    elif 'switch' in args.model : 
        normalization = SwitchNorm2d
    elif 'dbn' in args.model : 
        normalization = DBN
        normalization_params['num_groups'] = 16
        
    #Default Normalization is Batch Normalization 
    else : 
        normalization = nn.BatchNorm2d

    return normalization, normalization_params


def activation_building(args) : 
    if args.activation == 'relu' : activation = nn.ReLU
    elif args.activation == 'relu6' : activation = nn.ReLU6
    elif args.activation == 'leakyrelu' : activation = nn.LeakyReLU
    elif args.activation == 'prelu' : activation = nn.PReLU
    elif args.activation == 'mish' : activation = nn.Mish
    elif args.activation == 'silu' : activation = nn.SiLU
    elif args.activation == 'hardsilu' : activation = nn.Hardswish
    elif args.activation == 'la_silu' : activation = LA_SiLU
    elif args.activation == 'la_hardsilu' : activation = LA_HardSiLU
    elif args.activation == 'gelu' : activation = nn.GELU
    elif args.activation == 'elu' : activation = nn.ELU

    activation_params = {'alpha' : args.alpha} if 'la_' in args.activation else {}
    if 'la_' in args.activation and args.save_less : activation_params['save_less'] = True

    return activation, activation_params


def model_building(args, num_classes) : 
    activation, activation_params = activation_building(args)
    normalization, normalization_params = normaliation_building(args)

    if 'resnet' in args.model or 'resnext' in args.model : 
        if 'resnet18' in args.model : architecture = resnet18 
        elif 'resnet50' in args.model : architecture = resnet50
        elif 'resnet101' in args.model : architecture = resnet101
        elif 'resnet152' in args.model : architecture = resnet152
        elif 'resnet20' in args.model : architecture = resnet20
        elif 'resnet32' in args.model : architecture = resnet32
        elif 'resnet44' in args.model : architecture = resnet44
        elif 'resnext50-32x4d' in args.model : architecture = resnext50_32x4d

        return architecture(activation, activation_params, normalization, normalization_params, num_classes)

    elif 'unet' in args.model : 
        return UNet(activation, activation_params, 3, 1)

            