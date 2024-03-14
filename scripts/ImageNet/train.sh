#!/bin/bash

data=ImageNet
model=$1
activation=$2
data_path=$3
device=$4
alpha=0.1

device_ids='0,1'

learning_rate=0.1
batch_size=256

trial=1

pth_path='trained_networks/'
checkpoint_path='checkpoints/'

python3 ImageNet_resnet_train.py \
    -d $data -m $model -a $activation --data_path $data_path --device $device \
    --start_trial $trial --end_trial $trial \
    --alpha $alpha --learning_rate $learning_rate --batch_size $batch_size \
    --pth_path $pth_path --checkpoint_path $checkpoint_path \
    --device_ids $device_ids \