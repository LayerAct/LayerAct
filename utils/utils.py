import os 
import torch

__all__ = ['save_checkpoint', 'folder_check', 'update_print', 'noupdate_print']

def save_checkpoint(state, path, filename='checkpoint.pth.tar'):
    torch.save(state, path+filename)

def folder_check(path, name_list, make=False) : 
    try : os.makedirs(path)
    except : pass
    
    path_ = path + name_list[0] + '/'
    if make and name_list[0] not in os.listdir(path) : 
        os.makedirs(path_)
    if len(name_list) == 1 : 
        return path_ 
    else : 
        return folder_check(path_, name_list[1:], make)

def update_print(max_iter, iter, start_time, t, train_loss, train_acc1, train_acc5, val_loss, val_acc1, val_acc5, end='\r') : 
    try : 
        print(
            'Updated | Iter {}/{} | {}% | {} min | {} min left | Train loss {} | top1 {} | top5 {} | val loss {} | top1 {} | top5 {}'.format(
                iter, max_iter, round(100*(iter+1)/(max_iter)), 
                round((t-start_time)/60), round((t-start_time)/60*((max_iter-iter-1)/(iter+1))), 
                round(train_loss, 3), round(train_acc1.item(), 3), round(train_acc5.item(), 3),
                round(val_loss, 3), round(val_acc1.item(), 3), round(val_acc5.item(), 3)
                ) + ' '*30, end=end
            )
    except : 
        print(
            'Updated | Iter {}/{} | {}% | {} min | {} min left | Train loss {} | top1 {} | top5 {} | val loss {} | top1 {} | top5 {}'.format(
                iter, max_iter, round(100*(iter+1)/(max_iter)), 
                round((t-start_time)/60), round((t-start_time)/60*((max_iter-iter-1)/(iter+1))), 
                round(train_loss, 3), round(train_acc1, 3), round(train_acc5, 3),
                round(val_loss, 3), round(val_acc1, 3), round(val_acc5, 3)
                ) + ' '*30, end=end
            )
    

def noupdate_print(max_iter, iter, start_time, t, train_loss, train_acc1, train_acc5, val_loss, val_acc1, val_acc5, best_acc1, end='\r') : 
    try : 
        print(
            'Not | Iter {}/{} | {}% | {} min | {} min left | Train loss {} | top1 {} | top5 {} | Best val {} | val loss {} | top1 {} | top5 {}'.format(
                iter, max_iter, round(100*(iter+1)/(max_iter)), 
                round((t-start_time)/60), round((t-start_time)/60*((max_iter-iter-1)/(iter+1))), 
                round(train_loss, 3), round(train_acc1.item(), 3), round(train_acc5.item(), 3),
                round(best_acc1.item(), 3), 
                round(val_loss, 3), round(val_acc1.item(), 3), round(val_acc5.item(), 3)
                ) + ' '*30, 
                end=end
                #end='\n'
            )
    except : 
        print(
            'Not | Iter {}/{} | {}% | {} min | {} min left | Train loss {} | top1 {} | top5 {} | Best val {} | val loss {} | top1 {} | top5 {}'.format(
                iter, max_iter, round(100*(iter+1)/(max_iter)), 
                round((t-start_time)/60), round((t-start_time)/60*((max_iter-iter-1)/(iter+1))), 
                round(train_loss, 3), round(train_acc1, 3), round(train_acc5, 3),
                round(best_acc1, 3), 
                round(val_loss, 3), round(val_acc1, 3), round(val_acc5, 3)
                ) + ' '*30, 
                end=end
                #end='\n'
            )