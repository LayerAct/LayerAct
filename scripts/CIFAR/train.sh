#!/bin/bash

data=$1
model=$2
activation=$3
data_path=$4
device=$5
alpha=0.00001

learning_rate=0.1
batch_size=128

trial=1

pth_path='trained_networks/'
checkpoint_path='checkpoints/'

echo $data $model $activation $data_path

python3 cifar_resnet_train.py \
    -d $data -m $model -a $activation --data_path $data_path --device $device \
    --start_trial $trial --end_trial $trial \
    --alpha $alpha --learning_rate $learning_rate --batch_size $batch_size\
    --pth_path $pth_path --checkpoint_path $checkpoint_path