#!/bin/bash

data=$1
model=$2
activations=$3
data_path=$4
device=$5

batch_size=128

trial=1

model_path='trained_networks/'
save_path='results/'

loader=test
crop='center'

duplicate=True
batch_size=128

python3 resnet_test.py \
    --activations $activations --crop $crop --loader $loader --device $device \
    -d $data -m $model --data_path $data_path \
    --start_trial $trial --end_trial $trial \
    --model_path $model_path --save_path $save_path \
    --duplicate $duplicate --batch_size $batch_size