#!/bin/bash

data=unet
model=unet
activation=$1
data_path=$2
device=$3
alpha=0.00001

learning_rate=0.0001

trial=1

pth_path='trained_networks/'
checkpoint_path='checkpoints/'

python3 unet_train.py \
    -d $data -m $model -a $activation --data_path $data_path --device $device \
    --start_trial $trial --end_trial $trial \
    --alpha $alpha --learning_rate $learning_rate \
    --pth_path $pth_path --checkpoint_path $checkpoint_path
