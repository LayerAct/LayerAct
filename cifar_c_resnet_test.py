import argparse
import time
import os
import sys
import numpy as np
import pandas as pd
import random
import shutil
import math
from collections import OrderedDict as OD

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from data import image_data_building, corruption_image_data_building

from utils import save_checkpoint, folder_check, update_print, noupdate_print
from utils import train, validate, validate_10crop
from utils import cifar_c_testing_args_building

from models import model_building
    
random_seed = [11*i for i in range(1, 31)]

#######################################################################################################

if __name__ == '__main__' : 
    parser = argparse.ArgumentParser(description='')
    args = cifar_c_testing_args_building(parser)
    print(args)

    activation_name_list = []
    activation_list = []
    alpha_list = []
    for a in args.activations.split(',') : 
        activation_name_list.append(a)
        if 'alpha' in a : 
            activation_list.append('la_{}'.format(a.split('_')[1]))
            alpha_list.append(float(a.split('alpha')[-1]))
        else : 
            activation_list.append(a)
            alpha_list.append(0.00001)
    
    device = torch.device('cuda:{}'.format(args.device))
    model_path = folder_check(args.model_path, [args.data, args.model])
    save_path = folder_check(args.save_path, [args.data, args.model], True)

    duplicate = True if args.duplicate == 'True' else False

    path = args.data_path
    corruptions = args.corruptions.split(',')

    for count, corr in enumerate(corruptions) : 
        test_loader, out_num = corruption_image_data_building(args, path, corr)
        print('{} | {}/{}'.format(corr, count, len(corruptions)), end = '\n')

        for act_count, activation_name in enumerate(activation_name_list) : 
            args.activation = activation_list[act_count]
            args.alpha = alpha_list[act_count]
            current = {'loss' : [], 'acc1' : [], 'acc5' : []}
            if not duplicate and '{}_{}.csv'.format(corr, activation_name) in os.listdir(save_path) : 
                print("{} | {} exist ".format(args.model, activation_name), end = '\n')
            else : 
                for trial in range(args.start_trial, args.end_trial+1) : 
                    file_name = '{}_{}'.format(activation_name, trial)

                    rs = random_seed[trial-1]
                    random.seed(rs)
                    np.random.seed(rs)
                    torch.manual_seed(rs)
                    cudnn.deterministic = True
                    cudnn.benchmark = False

                    model = model_building(args, out_num)
                    model.to(device)
                    criterion = nn.CrossEntropyLoss().to(device)
                    
                    trained = torch.load(model_path + file_name + '.pth.tar', map_location=device)
                    try : 
                        model.load_state_dict(trained)
                    except : 
                        trained_ = OD([(k.split('module.')[-1], trained[k]) for k in trained.keys()])
                        model.load_state_dict(trained_)
                    if args.crop == '10crop' : 
                        test_loss, test_acc1, test_acc5 = validate_10crop(test_loader, model, criterion, device)
                    else : 
                        test_loss, test_acc1, test_acc5 = validate(test_loader, model, criterion, device)
                    print("{} | {} | {} | Test |  acc1 {} | acc5 {}".format(args.model, trial, activation_name, test_acc1, test_acc5), end = '\n')

                    if args.save : 
                        current['loss'].append(test_loss)
                        current['acc1'].append(test_acc1.item())
                        current['acc5'].append(test_acc5.item())
                
                indexes = [i for i in range(1, len(current['loss'])+1)] + ['mean', 'std']
                for k in current.keys() : 
                    current[k].append(np.mean(current[k]))
                    current[k].append(np.std(current[k]))

                pd.DataFrame(current, index=indexes).to_csv('{}{}_{}.csv'.format(save_path, corr, activation_name))
                print("{} | {} | {} done ".format(args.model, corr, activation_name), end = '\n')

