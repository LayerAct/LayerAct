# Layer-wise Balanced Activation Mechanism (LayerAct)
The source code and experiments of Layer-wise Balanced Activation Mechanism (LayerAct)
(https://openreview.net/forum?id=sqPEs1wEizU)

- [LayerAct.py](https://github.com/LayerAct/LayerAct/blob/main/models/LayerAct.py) is the source code of LA-SiLU and LA-HardSiLU implemented with pytorch.
- [ResNet.py](https://github.com/LayerAct/LayerAct/blob/main/models/ResNet.py) is our implementation of [ResNet (He et al., 2016)](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) based on the [official implementation of pytorch](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) for the experiment.  
- [trained_models](https://github.com/LayerAct/LayerAct/tree/main/trained_models) contains the trained networks of our experiments. 

## Experiment Environment
- python 3.9.12
- numpy 1.19.5
- pytorch 1.11.0
- torchvision 0.12.0

## Experiment Setting 
### CIFAR10 & CIFAR100
- Train/Validation: 55000/5000
- max iteration: around 64000 (until the last epoch end)
- learning rate: 0.1
- optimizer: Momentum (0.9) with weight decay 0.0001
- lr scheduler: 0.1 at iteration 32000 and 48000
- random seed for data split: 0
- random seed for weight initialization: 11*n (n=1,2,...)

## Experiment
- We show the performance of the models with the best and mean accuracy of 5-runs. 
- Top-1 accuracy: The model's prediction is correct when the prediction is exactly the same with the true label. 
- Top-5 accuracy: The model's prediction is correct when the top-5 scored predictions include the true label. 
### CIFAR10
- CIFAR10 is an image dataset for classification task with 10 classes. 
##### ResNet20
| Activation  | Top1 Best | Top1 Mean | 
| ----------- | --------- | --------- | 
|        ReLU |     91.68 |     91.42 | 
|        SiLU |     91.92 |     91.52 | 
|     LA-SiLU |     **91.97** |     **91.75** | 
| LA-HardSiLU |     91.74 |     91.47 | 

##### ResNet32
| Activation  | Top1 Best | Top1 Mean | 
| ----------- | --------- | --------- | 
|        ReLU |     91.96 |     91.79 | 
|        SiLU |     92.16 |     91.96 | 
|     LA-SiLU |     **92.64** |     **92.3** | 
| LA-HardSiLU |     92.16 |     91.84 | 

##### ResNet44
| Activation  | Top1 Best | Top1 Mean | 
| ----------- | --------- | --------- | 
|        ReLU |     92.59 |     92.38 | 
|        SiLU |     92.31 |     91.94 | 
|     LA-SiLU |     **93.00** |     **92.67** | 
| LA-HardSiLU |     91.93 |     91.68 | 


### CIFAR100
- CIFAR100 is an image dataset for classification task with 100 classes. 
##### ResNet20
| Activation  | Top1 Best | Top1 Mean | Top5 Best | Top5 Mean |   
| ----------- | --------- | --------- | --------- | --------- | 
|        ReLU |     66.25 |     65.98 |     89.81 |     89.62 | 
|        SiLU |     **66.68** |     66.09 |     90.23 |     89.59 | 
|     LA-SiLU |     66.65 |     **66.42** |     **90.32** |     **89.97** | 
| LA-HardSiLU |     66.64 |     66.30 |     90.04 |     89.84 | 

##### ResNet32
| Activation  | Top1 Best | Top1 Mean | Top5 Best | Top5 Mean | 
| ----------- | --------- | --------- | --------- | --------- | 
|        ReLU |     67.63 |     67.38 |     **90.34** |     90.09 | 
|        SiLU |     **68.17** |     67.45 |     90.05 |     89.92 | 
|     LA-SiLU |     67.99 |     **67.84** |     90.33 |     **90.11** | 
| LA-HardSiLU |     67.88 |     67.16 |     89.96 |     89.73 | 

##### ResNet44
| Activation  | Top1 Best | Top1 Mean | Top5 Best | Top5 Mean | 
| ----------- | --------- | --------- | --------- | --------- | 
|        ReLU |     68.68 |     68.44 |     90.47 |     90.29 | 
|        SiLU |     **68.99** |     68.37 |     90.55 |     90.04 | 
|     LA-SiLU |     68.89 |     **68.68** |     **90.80** |     **90.42** | 
| LA-HardSiLU |     67.67 |     66.80 |     89.60 |     89.34 | 

- More experiments will be updated. 
