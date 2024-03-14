import argparse
import time
import os
import sys
import numpy as np
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
from utils import cifar_training_args_building, ImageNet_training_args_building

from models import model_building
    
random_seed = [11*i for i in range(1, 31)]

#######################################################################################################

if __name__ == '__main__' : 
    parser = argparse.ArgumentParser(description='')
    args = ImageNet_training_args_building(parser)
    print(args)

    os.environ['CUDA_VISIBLE_DEVICE'] = args.device_ids

    learning_rate = args.learning_rate
    if 'lr' in args.model : 
        learning_rate = float(args.model.split['lr'][-1])

    milestones = [int(m) for m in args.milestones.split(',')]
    
    device_ids = [int(d) for d in args.device_ids.split(',')]
    output_device = torch.device('cuda:{}'.format(args.output_device))

    pth_path = folder_check(args.pth_path, [args.data, args.model], True)
    checkpoint_path = folder_check(args.checkpoint_path, [args.data, args.model], True)
    npy_path = folder_check(args.npy_path, [args.data, args.model], True)

    resume = True if args.resume == 'True' else False
    duplicate = True if args.duplicate == 'True' else False
    save = True if args.save == 'True' else False

    for trial in range(args.start_trial, args.end_trial+1) : 
        rs = random_seed[trial-1]
        random.seed(rs)
        np.random.seed(rs)
        torch.manual_seed(rs)
        cudnn.deterministic = True
        cudnn.benchmark = False

        train_loader, val_loader, test_loader, in_channel, H, W, out_num = image_data_building(args, rs)

        if 'la_' in args.activation and args.alpha != 1e-5 : 
            file_name = '{}_alpha{}_{}'.format(args.activation, args.alpha, trial)
        else : 
            file_name = '{}_{}'.format(args.activation, trial)

        if not duplicate and ('{}.pth.tar'.format(file_name) in os.listdir(pth_path) and '{}.npy'.format(file_name) in os.listdir(npy_path)) :
        #if not duplicate and '{}.pth.tar'.format(file_name) in os.listdir(save_path) :
            model_path = folder_check(args.pth_path, [args.data, args.model])
            model = model_building(args, out_num)
            model.to(torch.device('cuda'))
            criterion = nn.CrossEntropyLoss().to(torch.device('cuda'))
            
            trained = torch.load(model_path + file_name + '.pth.tar', map_location=torch.device('cuda'))
            try : 
                model.load_state_dict(trained)
            except : 
                trained_ = OD([(k.split('module.')[-1], trained[k]) for k in trained.keys()])
                model.load_state_dict(trained_)
            if args.crop == '10crop' : 
                test_loss, test_acc1, test_acc5 = validate_10crop(test_loader, model, criterion, torch.device('cuda'), output_device=output_device)
            else : 
                test_loss, test_acc1, test_acc5 = validate(test_loader, model, criterion, torch.device('cuda'), output_device=output_device)

            print('Model ({} | {} | {} | {}) exists | loss {} | top1 {} | top5 {} | '.format(
                args.data, args.model, args.activation, trial, round(test_loss, 4), round(test_acc1.item(), 2), round(test_acc5.item(), 2)))    
            
        else : 
            print(device_ids, output_device)
            model = model_building(args, out_num)
            model = nn.DataParallel(model, device_ids=device_ids, output_device=output_device).to(output_device)
            
            criterion = nn.CrossEntropyLoss().to(torch.device('cuda'))
            optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, last_epoch=0-1)

            print('model make', end='\r')
            best_model = None
            best_acc1 = 0
            start_time = time.time()
            start_iter = 0
            if resume and os.path.isfile(checkpoint_path + file_name + '_checkpoint.pth.tar') : 
                print('model resume', end='\r')
                checkpoint = torch.load(checkpoint_path + file_name + '_checkpoint.pth.tar', map_location=torch.device('cuda'))
                start_iter = checkpoint['iter']
                best_acc1 = checkpoint['best_acc1']

                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['scheduler'])
            
            np_save_dict = {
                'val_loss' : [], 'val_acc1' : [], 'val_acc5' : [], 
                'train_loss' : [], 'train_acc1' : [], 'train_acc5' : []
                }

            iter = start_iter
            while iter < args.max_iter : 
                iter, lr_scheduler = train(train_loader, model, criterion, optimizer, lr_scheduler, output_device, iter, output_device=output_device)
                val_loss, val_acc1, val_acc5 = validate(val_loader, model, criterion, device=torch.device('cuda'), output_device=output_device)
                train_loss, train_acc1, train_acc5 = validate(train_loader, model, criterion, device=torch.device('cuda'), output_device=output_device)
            
                if args.save : 
                    np_save_dict['val_loss'].append(val_loss)
                    np_save_dict['val_acc1'].append(val_acc1.item())
                    np_save_dict['val_acc5'].append(val_acc5.item())
                    np_save_dict['train_loss'].append(train_loss)
                    np_save_dict['train_acc1'].append(train_acc1.item())
                    np_save_dict['train_acc5'].append(train_acc5.item())

                t = time.time()
                is_best = val_acc1 > best_acc1 
                best_acc1 = max(val_acc1, best_acc1)

                if is_best : 
                    best_model = model.state_dict()
                    best_iter = iter
                    update_print(args.max_iter, iter, start_time, t, train_loss, train_acc1, train_acc5, val_loss, val_acc1, val_acc5, '\n')
                    
                else : 
                    noupdate_print(args.max_iter, iter, start_time, t, train_loss, train_acc1, train_acc5, val_loss, val_acc1, val_acc5, best_acc1, '\n')
                
                if args.save : 
                    save_checkpoint(
                        {
                            'iter' : iter + 1, 
                            'time' : t,
                            'state_dict' : model.state_dict(),
                            'best_model' : best_model,
                            'best_acc1' : best_acc1, 
                            'optimizer' : optimizer.state_dict(), 
                            'scheduler' : lr_scheduler.state_dict(), 
                        }, checkpoint_path, file_name + '_checkpoint.pth.tar'
                    )
                
                if iter > args.max_iter : 
                    break
            
            if args.save : 
                np.save('{}.npy'.format(npy_path+file_name), np_save_dict, allow_pickle=True)
                torch.save(best_model, '{}.pth.tar'.format(pth_path + file_name))

            model.load_state_dict(best_model)
            test_loss, test_acc1, test_acc5 = validate(test_loader, model, criterion, device=output_device)

            print("{} | {} | {} | Test |  acc1 {} | acc5 {} ".format(
                args.model, trial, args.activation, round(test_acc1.item(), 3), round(test_acc5.item(), 3)) + ' '*100, end = '\n')

