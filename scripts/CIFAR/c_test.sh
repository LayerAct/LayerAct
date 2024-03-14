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

corruptions='gaussian_noise,shot_noise,impulse_noise,defocus_blur,motion_blur,zoom_blur,glass_blur,jpeg_compression,elastic_transform,pixelate,contrast,fog,glass_blur,jpeg_compression,elastic_transform,pixelate,contrast,fog'

python3 cifar_c_resnet_test.py \
    -d $data -m $model --device $device \
    --start_trial $trial --end_trial $trial \
    --corruptions $corruptions --activations $activations \
    --data_path $data_path --model_path $model_path --save_path $save_path --duplicate $duplicate