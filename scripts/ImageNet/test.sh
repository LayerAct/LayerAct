#!/bin/bash

data=ImageNet
model=$1
activations=$2
data_path=$3
device=$4

trial=1

model_path='trained_networks/'
save_path='results/'

loader=test
crop='10crop'

duplicate=True
batch_size=128

python3 resnet_test.py \
    --activations $activations --crop $crop --loader $loader --device $device \
    -d $data -m $model --data_path $data_path \
    --start_trial $trial --end_trial $trial \
    --model_path $model_path --save_path $save_path \
    --duplicate $duplicate --batch_size $batch_size