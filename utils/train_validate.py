import time
from enum import Enum

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as dst
import torchvision.transforms as transforms
from torch.autograd import Variable as V

#from .attacks import select_attack

import shutil
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def train(train_loader, model, criterion, optimizer, lr_scheduler, device, iter, output_device=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    if output_device is None : 
        output_device = device
    else : 
        if type(output_device) == int : 
            output_device = torch.device('cuda:{}'.format(output_device))
        else : 
            output_device = output_device

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        #print('training iter {}'.format(i) + ' '*30, end='\r')
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(output_device, non_blocking=True)
        #print('training iter {} to'.format(i) + ' '*30, end='\r')

        #print(target)

        output = model(images)
        #print('training iter {} output'.format(i) + ' '*30, end='\r')
        try : 
            loss = criterion(output, target)
        except : 
            loss = criterion(output, target.type(torch.int64))
        #print('training iter {} loss'.format(i) + ' '*30, end='\r')

        #acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        #top1.update(acc1[0], images.size(0))
        #top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        #print('training iter {} backward'.format(i) + ' '*30, end='\r')
        optimizer.step()
        if lr_scheduler is not None : 
            lr_scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()
        iter += 1
        
    return iter, lr_scheduler


def train_epoch(train_loader, model, criterion, optimizer, device, output_device=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    
    if output_device is None : 
        output_device = device
    else : 
        if type(output_device) == int : 
            output_device = torch.device('cuda:{}'.format(output_device))
        else : 
            output_device = output_device

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        #print('training iter {}'.format(i) + ' '*30, end='\r')
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(output_device, non_blocking=True)
        #print('training iter {} to'.format(i) + ' '*30, end='\r')

        #print(target)

        output = model(images)
        #print('training iter {} output'.format(i) + ' '*30, end='\r')
        try : 
            loss = criterion(output, target)
        except : 
            loss = criterion(output, target.type(torch.int64))
        #print('training iter {} loss'.format(i) + ' '*30, end='\r')

        #acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        #top1.update(acc1[0], images.size(0))
        #top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        #print('training iter {} backward'.format(i) + ' '*30, end='\r')
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
    return 


def analysis_train(
        train_loader, model, criterion, optimizer, lr_scheduler, device, iter, trial, save_path, output_device=None
        ):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    if output_device is None : 
        output_device = device
    else : 
        output_device = torch.device('cuda:{}'.format(output_device))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        print(i, end='\r')
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
                
        images = images.to(device, non_blocking=True)
        target = target.to(output_device, non_blocking=True)

        output = images.view(images.shape[0], -1)
        output = model.linear1(output)
        torch.save(output, save_path + '{}_{}_input.pth.tar'.format(trial, iter))
        output = model.activation(output)
        torch.save(output, save_path + '{}_{}_output.pth.tar'.format(trial, iter))
        output = model.linear2(output)

        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        #acc1, acc5 = accuracy(output, target, topk=(1, 5))
        #losses.update(loss.item(), images.size(0))
        #top1.update(acc1[0], images.size(0))
        #top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        iter += 1
        
    return iter, lr_scheduler

def validate(val_loader, model, criterion, device, output_device=None):
    
    if output_device is None : 
        output_device = device
    else : 
        if type(output_device) == int : 
            output_device = torch.device('cuda:{}'.format(output_device))
        else : 
            output_device = output_device

    def run_validate(loader, base_progress=0, topk=(1,5)):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                #print('validation iter {}'.format(i) + ' '*20, end='\r')
                
                images = images.to(device, non_blocking=True)
                target = target.to(output_device, non_blocking=True)

                output = model(images)
                try : 
                  loss = criterion(output, target)
                except : 
                  print('i : ', i, ' | output : ', output.device, ' | target : ', target.device)

                acc1, acc5 = accuracy(output, target, topk=topk)
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)

    return losses.avg, top1.avg, top5.avg



class seg_AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seg_train(train_loader, model, grad_scaler, criterion, metric, optimizer, device):
    avg_meters = {'loss': seg_AverageMeter(),
                  'metric_loss': seg_AverageMeter()}

    model.train()

    #for input, target, _ in train_loader:
    for input, target in train_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(input)

        #print(input.shape, output.shape, target.shape)
        #B
        loss = criterion(output.squeeze(1), target.squeeze(1).float())
        metric_loss = metric(output.squeeze(1), target.squeeze(1).float())
        loss += metric_loss

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        #loss.backward()
        #optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['metric_loss'].update(metric_loss, input.size(0))

    return avg_meters['loss'].avg, avg_meters['metric_loss'].avg


from .m_dice_loss import dice_coeff

def seg_validate(val_loader, model, criterion, metric, device):
    
    losses = seg_AverageMeter()

    # switch to evaluate mode
    model.eval()
    dice_score = 0
    with torch.no_grad():
        #for input, target, _ in val_loader:
        for input, target in val_loader:
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(input)

            loss = criterion(output.squeeze(1), target.squeeze(1).float())
            metric_loss = metric(output.squeeze(1), target.squeeze(1).float())
            loss += metric_loss

            losses.update(loss.item(), input.size(0))

            output = (F.sigmoid(output) > 0.5).float()
            dice_score += dice_coeff(output, target, reduce_batch_first=False)

    return losses.avg, dice_score/max(len(val_loader), 1)


def validate_10crop(val_loader, model, criterion, device):
    def run_validate(loader, base_progress=0, topk=(1,5)):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if device is not None and torch.cuda.is_available():
                    images = images.cuda(device, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(device, non_blocking=True)
                
                bs, ncrops, c, h, w = images.size()
                images = images.view(-1, c, h, w)

                # compute output
                output = model(images)
                output = output.view(bs, ncrops, -1).mean(1)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=topk)
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)

    return losses.avg, top1.avg, top5.avg


def validate_ImageNet_C(model, criterion, device, corr, path, batch_size, num_workers, output_device=None):
    
    if output_device is None  : 
        output_device = device
    else : 
        if type(output_device) == int : 
            output_device = torch.device('cuda:{}'.format(output_device))
        else : 
            output_device = output_device
    
    def run_validate(base_progress=0, topk=(1,5)):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
        
        for severity in range(1, 6) : 

            batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
            current_losses = AverageMeter('Loss', ':.4e', Summary.NONE)
            current_top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
            current_top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

            distorted_dataset = dst.ImageFolder(
                root=path+'/'+corr+'/'+str(severity), 
                transform=transforms.Compose([transforms.ToTensor(), normalize])
            )

            loader = torch.utils.data.DataLoader(
                distorted_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
            )

            with torch.no_grad():
                end = time.time()
                for i, (images, target) in enumerate(loader):
                    i = base_progress + i
                    
                    #images = images.to(device, non_blocking=True)
                    images = V(images.to(device, non_blocking=True), volatile=True)
                    target = target.to(output_device, non_blocking=True)
                
                    output = model(images)
                    try : 
                        loss = criterion(output, target)
                    except : 
                        print('i : ', i, ' | output : ', output.device, ' | target : ', target.device)

                    acc1, acc5 = accuracy(output, target, topk=topk)
                    current_losses.update(loss.item(), images.size(0))
                    current_top1.update(acc1[0], images.size(0))
                    current_top5.update(acc5[0], images.size(0))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

            #print(current_losses.avg)
            #print(current_top1.avg.item())
            #print(current_top5.avg.item())            

            losses.append(current_losses.avg)
            top1.append(current_top1.avg.item())
            top5.append(current_top5.avg.item())

    losses = []
    top1 = []
    top5 = []

    # switch to evaluate mode
    model.eval()

    run_validate()

    return np.mean(losses), np.mean(top1), np.mean(top5)
    #return losses.avg, top1.avg, top5.avg
