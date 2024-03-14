import argparse

def cifar_training_args_building(parser) : 
    parser.add_argument('--data', '-d', default='CIFAR10', type=str)
    parser.add_argument('--model', '-m', default='resnet20', type=str)
    parser.add_argument('--activation', '-a', default='relu', type=str)

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--crop', default='center', type=str)
    parser.add_argument('--start_trial', default=1, type=int)
    parser.add_argument('--end_trial', default=30, type=int)

    parser.add_argument('--alpha', default=1e-5, type=float)
    parser.add_argument('--save_less', default=False, type=bool)
    parser.add_argument('--batch_size', '-bs', default=128, type=int)
    parser.add_argument('--num_workers', '-nw', default=16, type=int)
    parser.add_argument('--learning_rate', '-lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight_decay', '-wd', default=1e-4, type=float)    
    parser.add_argument('--max_iter', default=64000, type=int)
    parser.add_argument('--milestones', default='32000,48000', type=str)

    parser.add_argument('--data_path', '-dp', default='', type=str)
    parser.add_argument('--pth_path', default='../../trained_nets/', type=str)
    parser.add_argument('--checkpoint_path', default = '../../checkpoints/', type=str)
    parser.add_argument('--npy_path', default='../../npy_files/', type=str)
    parser.add_argument('--resume', default="True", type=str)
    parser.add_argument('--duplicate', default="False", type=str)
    parser.add_argument('--save', default="True", type=str)

    args = parser.parse_args()

    return args


def cifar_testing_args_building(parser) : 

    parser.add_argument('--data', '-d', default='CIFAR10')
    parser.add_argument('--model', '-m', default='resnet20')
    parser.add_argument('--activations', '-a', default='relu,leakyrelu,prelu,mish,silu,hardsilu,gelu,elu,la_silu,la_hardsilu')
    parser.add_argument('--save_less', default=False, type=bool)
    parser.add_argument('--loader', default='test', type=str)

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--crop', default='center')
    parser.add_argument('--start_trial', default=1, type=int)
    parser.add_argument('--end_trial', default=5, type=int)

    parser.add_argument('--batch_size', '-bs', default=128, type=int)
    parser.add_argument('--num_workers', '-nw', default=16)

    parser.add_argument('--data_path', '-dp', default='')
    parser.add_argument('--model_path', default='../../trained_nets/')
    parser.add_argument('--save_path', default='../../results/')

    parser.add_argument('--resume', default=True, type=bool)
    parser.add_argument('--duplicate', default="False", type=str)
    parser.add_argument('--save', default="True", type=str)
    
    args = parser.parse_args()

    return args

def cifar_c_testing_args_building(parser) : 

    parser.add_argument('--data', '-d', default='CIFAR10')
    parser.add_argument('--model', '-m', default='resnet20')
    parser.add_argument('--activations', '-a', default='relu,leakyrelu,prelu,mish,silu,hardsilu,gelu,elu,la_silu,la_hardsilu')
    parser.add_argument('--alpha', default=1e-5, type=float)
    parser.add_argument('--save_less', default=False, type=bool)

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--crop', default='center')
    parser.add_argument('--start_trial', default=1, type=int)
    parser.add_argument('--end_trial', default=5, type=int)

    parser.add_argument('--batch_size', '-bs', default=128)
    parser.add_argument('--num_workers', '-nw', default=16)

    parser.add_argument('--data_path', '-dp', default='')
    parser.add_argument('--model_path', default='../../trained_nets/')
    parser.add_argument('--save_path', default='../../results/')

    parser.add_argument('--resume', default=True, type=bool)
    parser.add_argument('--duplicate', default="False", type=str)
    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument(
        '--corruptions', 
        default='gaussian_noise,shot_noise,impulse_noise,defocus_blur,motion_blur,zoom_blur,glass_blur,jpeg_compression,elastic_transform,pixelate,contrast,fog,frost,snow,brightness,speckle,gaussian_blur,spatter,saturate',
        type=str
        )
    
    parser.add_argument('--adap_batch_size', default=32, type=int)
    
    args = parser.parse_args()

    return args

def ImageNet_training_args_building(parser) : 

    parser.add_argument('--data', '-d', default='ImageNet')
    parser.add_argument('--model', '-m', default='resnet50')
    parser.add_argument('--activation', '-a', default='relu', type=str)
    parser.add_argument('--save_less', default=False, type=bool)

    parser.add_argument('--device_ids', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--output_device', default='0')
    parser.add_argument('--crop', default='center', type=str)
    parser.add_argument('--start_trial', default=1, type=int)
    parser.add_argument('--end_trial', default=1, type=int)

    parser.add_argument('--alpha', default=1e-5, type=float)
    parser.add_argument('--batch_size', '-bs', default=256)
    parser.add_argument('--num_workers', '-nw', default=16)
    parser.add_argument('--learning_rate', '-lr', default=0.1)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight_decay', '-wd', default=0.0001)    
    parser.add_argument('--max_iter', default=600000)
    parser.add_argument('--milestones', default='180000,360000,540000')

    parser.add_argument('--data_path', '-dp', default='', type=str)
    parser.add_argument('--pth_path', default='../../trained_nets/', type=str)
    parser.add_argument('--checkpoint_path', default = '../../checkpoints/', type=str)
    parser.add_argument('--npy_path', default='../../npy_files/', type=str)
    parser.add_argument('--resume', default="True", type=str)
    parser.add_argument('--duplicate', default="False", type=str)
    parser.add_argument('--save', default="True", type=str)
    
    args = parser.parse_args()

    return args


def unet_training_args_building(parser) : 

    parser.add_argument('--data', '-d', default='unet')
    parser.add_argument('--model', '-m', default='unet')
    parser.add_argument('--activation', '-a', default='relu', type=str)
    parser.add_argument('--save_less', default=False, type=bool)

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--start_trial', default=1, type=int)
    parser.add_argument('--end_trial', default=5, type=int)

    parser.add_argument('--alpha', default=1e-5, type=float)
    parser.add_argument('--batch_size', '-bs', default=16, type=int)
    parser.add_argument('--num_workers', '-nw', default=4, type=int)
    parser.add_argument('--learning_rate', '-lr', default=0.0001, type=float)
    parser.add_argument('--max_epoch', default=200, type=int)
    parser.add_argument('--milestones', default='100,150')

    parser.add_argument('--data_path', '-dp', default='../../../data/archive/kaggle_3m/', type=str)
    parser.add_argument('--pth_path', default='../../trained_nets/', type=str)
    parser.add_argument('--checkpoint_path', default = '../../checkpoints/', type=str)
    parser.add_argument('--npy_path', default='../../npy_files/', type=str)
    parser.add_argument('--resume', default="True", type=str)
    parser.add_argument('--duplicate', default="False", type=str)
    parser.add_argument('--save', default="True", type=str)
    
    args = parser.parse_args()

    return args