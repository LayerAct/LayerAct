import os
import random
import numpy as np
import pandas as pd
from glob import glob

from sklearn.model_selection import train_test_split

import torch.utils.data
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from PIL import Image


class CustomDataset(Dataset) : 
    def __init__(self, image_paths, target_paths, train=True):
        self.image_paths = image_paths
        self.target_paths = target_paths

    def transform(self, image, mask):
        # Resize
        resize = transforms.Resize(size=(256, 256))
        image = resize(image)
        mask = resize(mask)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        return image, mask

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])
        x, y = self.transform(image, mask)
        return x, y

    def __len__(self):
        return len(self.image_paths)

def load_unet(data_path, batch_size, num_workers, random_seed) : 

    train_files = []
    mask_files = glob(data_path+'*/*_mask*')

    for i in mask_files:
        train_files.append(i.replace('_mask',''))

    #Create dataframes with paths for training, validation, and testing
    df = pd.DataFrame(data={"filename": train_files, 'mask' : mask_files})
    df_train, df_test = train_test_split(df,test_size=0.1, random_state=random_seed)
    df_train, df_val = train_test_split(df_train,test_size = 0.2, random_state=random_seed)

    #Datasets
    train_dataset = CustomDataset(df_train["filename"].values.tolist(), df_train["mask"].values.tolist())
    val_dataset = CustomDataset(df_val['filename'].values.tolist(), df_val['mask'].values.tolist())
    test_dataset = CustomDataset(df_test['filename'].values.tolist(), df_test["mask"].values.tolist())

    #Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader