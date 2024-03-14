import os
import numpy as np
import torch.utils.data

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
import random


def CIFAR_transforms(normalize, test, random_seed) : 
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    compose_list = [transforms.ToTensor()]

    if not test : 
        compose_list += [transforms.RandomHorizontalFlip(),  transforms.RandomCrop(32, 4)]
    compose_list.append(normalize)

    return transforms.Compose(compose_list)


def load_MNIST(data_path, batch_size, num_workers, random_seed) : 
    normalize = transforms.Normalize(mean=0.5,
                                     std=0.5)
    trans = transforms.Compose([
            transforms.ToTensor(),
            normalize,        
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root = data_path, train=True, transform=trans, download=False
    )
    test_dataset = torchvision.datasets.MNIST(
        root = data_path, train=False, transform=trans, download=False
    )
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_seed)
    indices = list(range(len(train_dataset)))
    train_list = [t for _, t in train_dataset]

    for train_index, val_index in sss.split(indices, train_list):
        train_index = train_index
        val_index = val_index
        
    train_sampler = SubsetRandomSampler(train_index)
    val_sampler = SubsetRandomSampler(val_index)
    
    pin_memory = True

    g = torch.Generator()
    g.manual_seed(0)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, 
        num_workers=num_workers, pin_memory = pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=val_sampler, 
        num_workers=num_workers, pin_memory = pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, 
        num_workers=num_workers, pin_memory = pin_memory
    )
    
    return train_loader, val_loader, test_loader


def load_CIFAR10(data_path, batch_size, num_workers, random_seed) : 
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    train_dataset = torchvision.datasets.CIFAR10(root = data_path, train=True, transform=transforms.ToTensor(), download=False)
    test_dataset = torchvision.datasets.CIFAR10(root = data_path, train=False, transform=transforms.ToTensor(), download=False)

    imgs = torch.stack([d[0] for d in train_dataset], dim=0).numpy()

    mean = [imgs[:, 0, :, :].mean(), imgs[:, 1, :, :].mean(), imgs[:, 2, :, :].mean()]
    std = [imgs[:, 0, :, :].std(), imgs[:, 1, :, :].std(), imgs[:, 2, :, :].std()]

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transforms = CIFAR_transforms(normalize, False, random_seed)
    test_transforms = CIFAR_transforms(normalize, True, random_seed)
    
    train_dataset = torchvision.datasets.CIFAR10(root = data_path, train=True, transform=train_transforms, download=False)
    test_dataset = torchvision.datasets.CIFAR10(root = data_path, train=False, transform=test_transforms, download=False)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_seed)
    indices = list(range(len(train_dataset)))
    train_list = [t for _, t in train_dataset]

    for train_index, val_index in sss.split(indices, train_list):
        train_index = train_index
        val_index = val_index
        
    train_sampler = SubsetRandomSampler(train_index)
    val_sampler = SubsetRandomSampler(val_index)

    pin_memory = True
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, 
        num_workers=num_workers, pin_memory = pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=val_sampler, 
        num_workers=num_workers, pin_memory = pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, 
        num_workers=num_workers, pin_memory = pin_memory
    )
    
    return train_loader, val_loader, test_loader


def load_CIFAR100(data_path, batch_size, num_workers, random_seed) : 
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    train_dataset = torchvision.datasets.CIFAR100(root = data_path, train=True, transform=transforms.ToTensor(), download=False)
    test_dataset = torchvision.datasets.CIFAR100(root = data_path, train=False, transform=transforms.ToTensor(), download=False)

    imgs = torch.stack([d[0] for d in train_dataset], dim=0).numpy()

    mean = [imgs[:, 0, :, :].mean(), imgs[:, 1, :, :].mean(), imgs[:, 2, :, :].mean()]
    std = [imgs[:, 0, :, :].std(), imgs[:, 1, :, :].std(), imgs[:, 2, :, :].std()]

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transforms = CIFAR_transforms(normalize, False, random_seed)
    test_transforms = CIFAR_transforms(normalize, True, random_seed)
    
    train_dataset = torchvision.datasets.CIFAR100(root = data_path, train=True, transform=train_transforms, download=False)
    test_dataset = torchvision.datasets.CIFAR100(root = data_path, train=False, transform=test_transforms, download=False)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_seed)
    indices = list(range(len(train_dataset)))
    train_list = [t for _, t in train_dataset]

    for train_index, val_index in sss.split(indices, train_list):
        train_index = train_index
        val_index = val_index
        
    train_sampler = SubsetRandomSampler(train_index)
    val_sampler = SubsetRandomSampler(val_index)
    
    pin_memory = True
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, 
        num_workers=num_workers, pin_memory = pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=val_sampler, 
        num_workers=num_workers, pin_memory = pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, 
        num_workers=num_workers, pin_memory = pin_memory
    )
    
    return train_loader, val_loader, test_loader


def imagenet_transforms(normalize, test, crop='center', random_seed=0) : 
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    compose_list = [transforms.ToTensor()]

    if not test : 
        compose_list += [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), normalize]
    else : 
        if crop == 'random' : 
            compose_list += [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), normalize]
        elif crop == 'center' : 
            compose_list += [transforms.Resize(256), transforms.CenterCrop(224), normalize]
        elif crop == '10crop' : 
            compose_list = [
                transforms.Resize(256), transforms.TenCrop(224), 
                transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])), 
            ] 
            
    return transforms.Compose(compose_list)    
    

def load_ImageNet(data_path, batch_size, num_workers, random_seed, crop) :
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )

    train_transforms = imagenet_transforms(normalize, False, random_seed=random_seed)
    test_transforms = imagenet_transforms(normalize, True, crop, random_seed=random_seed)
    
    if crop == '10crop' : 
        batch_size = 32
    else : 
        batch_size = 256
    pin_memory = True

    train_dataset = torchvision.datasets.ImageFolder(root = data_path + '/train', transform=train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(root = data_path + '/val', transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory = pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, 
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, val_loader, val_loader
