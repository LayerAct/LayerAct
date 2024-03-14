#!/bin/bash

data=ImageNet
model=$1
activations=$2
data_path=$3
device=$4

device_ids='0,1'
alpha=0.1

trial=1

model_path='trained_networks/'
save_path='result/'

crop='center'

duplicate=True

python3 imagenet_c_resnet_test.py \
    --activations $activations --crop $crop --device $device \
    -d $data -m $model --data_path $data_path \
    --start_trial $trial --end_trial $trial \
    --model_path $model_path --save_path $save_path \
    --duplicate $duplicate --alpha $alpha