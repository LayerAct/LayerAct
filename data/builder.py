
from .image_data_augmentation import load_CIFAR10, load_CIFAR100, load_ImageNet, load_MNIST
from .unet_data_augmentation import load_unet
from .c_data_loader import CIFAR10C, CIFAR100C

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def image_data_building(args, rs) :
    if 'resnet' in args.model : 
        if args.data == 'CIFAR10' : 
            train_loader, val_loader, test_loader = load_CIFAR10(args.data_path, args.batch_size, args.num_workers, rs)
            in_channel, H, W, out_num = 3, 32, 32, 10 
        elif args.data == 'CIFAR100' : 
            train_loader, val_loader, test_loader = load_CIFAR100(args.data_path, args.batch_size, args.num_workers, rs)
            in_channel, H, W, out_num = 3, 32, 32, 100 
        elif args.data == 'MNIST' : 
            train_loader, val_loader, test_loader = load_MNIST(args.data_path, args.batch_size, args.num_workers, rs)
            in_channel, H, W, out_num = 1, 28, 28, 100 
        elif args.data == 'ImageNet' : 
            train_loader, val_loader, test_loader = load_ImageNet(args.data_path, args.batch_size, args.num_workers, rs, args.crop)
            in_channel, H, W, out_num = 3, 224, 224, 1000 
        else : 
            raise Exception('Dataset should be "CIFAR10", "CIFAR100", and "ImageNet"') 
        
    if args.data == 'unet' and args.model == 'unet' : 
        train_loader, val_loader, test_loader = load_unet(args.data_path, args.batch_size, args.num_workers, rs)
        in_channel, H, W, out_num = 3, 256, 256, 1
    
    return train_loader, val_loader, test_loader, in_channel, H, W, out_num


def corruption_image_data_building(args, path, corr) : 
    if args.data == 'CIFAR10' : 
        MEAN = [0.49139968, 0.48215841, 0.44653091]
        STD  = [0.24703223, 0.24348513, 0.26158784]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        dataset = CIFAR10C(path, corr, transform=transform)
        
        test_loader = DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=4
            )
        out_num=10
        
    elif args.data == 'CIFAR100' : 
        MEAN = [0.5071, 0.4867, 0.4408]
        STD  = [0.2675, 0.2565, 0.2761]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        dataset = CIFAR100C(path, corr, transform=transform)
        
        test_loader = DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=4
            )
        out_num=100

    return test_loader, out_num
        
