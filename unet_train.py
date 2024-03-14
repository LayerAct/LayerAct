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
from utils import train, validate, validate_10crop, seg_train, seg_validate
from utils import unet_training_args_building
from utils import BCEDiceLoss, iou_score, dice_loss


from models import model_building
    
random_seed = [11*i for i in range(1, 31)]

#######################################################################################################

if __name__ == '__main__' : 
    parser = argparse.ArgumentParser(description='')
    args = unet_training_args_building(parser)
    print(args)

    learning_rate = args.learning_rate
    if 'lr' in args.model : 
        learning_rate = float(args.model.split['lr'][-1])

    milestones = [int(m) for m in args.milestones.split(',')]
    
    device = torch.device('cuda:{}'.format(args.device))
    pth_path = folder_check(args.pth_path, [args.data, args.model], True)
    checkpoint_path = folder_check(args.checkpoint_path, [args.data, args.model], True)
    npy_path = folder_check(args.npy_path, [args.data, args.model], True)

    resume = True if args.resume == 'True' else False
    resume = False
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
            model.to(device)
            criterion = BCEDiceLoss()
            #metric = JaccardIndex(task='binary', num_classes=out_num)
            metric = iou_score
            
            trained = torch.load(model_path + file_name + '.pth.tar', map_location=device)
            try : 
                model.load_state_dict(trained)
            except : 
                trained_ = OD([(k.split('module.')[-1], trained[k]) for k in trained.keys()])
                model.load_state_dict(trained_)
            if args.crop == '10crop' : 
                test_loss, test_metric_loss, test_acc5 = validate_10crop(test_loader, model, criterion, device)
            else : 
                test_loss, test_metric_loss = seg_validate(test_loader, model, criterion, metric, device)

            print('Model ({} | {} | {} | {}) exists | loss {} | top1 {} | top5 {} | '.format(
                args.data, args.model, args.activation, trial, round(test_loss, 4), round(test_metric_loss.item(), 2), round(test_acc5.item(), 2)))    
            
        else : 
            model = model_building(args, out_num)
            model.to(device)
            
            #criterion = DiceLoss()
            #criterion = SoftDiceLoss()
            grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
            #criterion = BCEDiceLoss()
            criterion = nn.BCEWithLogitsLoss()
            #metric = JaccardIndex(task='binary', num_classes=out_num)
            metric = dice_loss

            optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, last_epoch=0-1)
            
            np_save_dict = {
                'val_loss' : [], 'val_score' : [], 
                'train_loss' : [], 'train_metric_loss' : [], 
                }

            print('model make', end='\r')
            best_model = None
            best_score = 1e+100
            best_metric_loss = 0
            start_time = time.time()
            start_epoch = 0

            epoch = start_epoch
            while epoch < args.max_epoch : 
                model.train()
                train_loss, train_metric_loss = seg_train(train_loader, model, grad_scaler, criterion, metric, optimizer, device)
                model.eval()
                train_loss, train_score = seg_validate(train_loader, model, criterion, metric, device)
                val_loss, val_score = seg_validate(val_loader, model, criterion, metric, device)
            
                if args.save : 
                    np_save_dict['val_loss'].append(val_loss)
                    np_save_dict['val_score'].append(val_score)
                    np_save_dict['train_loss'].append(train_loss)
                    np_save_dict['train_metric_loss'].append(train_metric_loss)

                t = time.time()
                #is_best = val_score > best_metric_loss
                #best_metric_loss = max(val_score, best_metric_loss)
                is_best = val_score.item() > best_score
                best_score = min(val_score.item(), best_score)

                if is_best : 
                    best_model = model.state_dict()
                    best_epoch = epoch

                print(
                    'epoch {}/{} | {}% | {} min | {} min left | Train loss {} | score {} | val loss {} | score {} '.format(
                        epoch, args.max_epoch, round(100*(epoch+1)/(args.max_epoch)), 
                        round((t-start_time)/60), round((t-start_time)/60*((args.max_epoch-epoch-1)/(epoch+1))), 
                        round(train_loss, 3), round(train_score.item(), 3),
                        round(val_loss, 3), round(val_score.item(), 3)
                        ) + ' '*30, end='\r'
                    )
                
                if args.save : 
                    save_checkpoint(
                        {
                            'epoch' : epoch + 1, 
                            'time' : t,
                            'state_dict' : model.state_dict(),
                            'best_model' : best_model,
                            'best_metric_loss' : best_metric_loss, 
                            'optimizer' : optimizer.state_dict(), 
                        }, checkpoint_path, file_name + '_checkpoint.pth.tar'
                    )

                lr_scheduler.step()
                epoch += 1
            
            if args.save : 
                np.save('{}.npy'.format(npy_path+file_name), np_save_dict, allow_pickle=True)
                torch.save(best_model, '{}.pth.tar'.format(pth_path + file_name))

            model.load_state_dict(best_model)
            test_loss, test_score = seg_validate(test_loader, model, criterion, metric, device)

            print("{} | {} | {} | Test | loss {} | score {} ".format(
                args.model, trial, args.activation, round(test_loss, 3), round(test_score.item(), 3)) + ' '*100, end = '\n')

