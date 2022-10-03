# Layer-wise Balanced Activation Mechanism (LayerAct)
The source code of Layer-wise Balanced Activation Mechanism
(https://openreview.net/forum?id=sqPEs1wEizU)

## Experiment Environment
- python 3.9.12
- numpy 1.19.5
- pytorch 1.11.0
- torchvision 0.12.0

## Experiment Setting 
- max iteration: around 64000 (until the last epoch end)
- learning rate: 0.1
- optimizer: Momentum (0.9) with weight decay 0.0001
- lr scheduler: 0.1 at iteration 32000 and 48000
- random seed for data split: 0
- random seed for weight initialization: 11*n (n=1,2,...)

## Accuracy 
### CIFAR10
#### ResNet20
| Activation | Top1 Best | Top1 Mean | 
| ----------- | --------- | --------- | 
|        ReLU |     91.68 |     91.42 | 
|        SiLU |     91.92 |     91.52 | 
|     LA-SiLU |     **91.97** |     **91.75** | 

#### ResNet32
| Activation | Top1 Best | Top1 Mean | 
| ----------- | --------- | --------- | 
| LA-HardSiLU |     91.74 |     91.47 | 
|        ReLU |     91.96 |     91.79 | 
|        SiLU |     92.16 |     91.96 | 
|     LA-SiLU |     **92.64** |     **92.3** | 
| LA-HardSiLU |     92.16 |     91.84 | 

### ResNet44
|        ReLU |     92.59 |     92.38 | 
|        SiLU |     92.31 |     91.94 | 
|     LA-SiLU |     **93.00** |     **92.67** | 
| LA-HardSiLU |     91.93 |     91.68 | 

### CIFAR100
#### ResNet20
| Activation | Top1 Best | Top1 Mean | 
| ----------- | --------- | --------- | 
|        ReLU |     91.68 |     91.42 | 
|        SiLU |     91.92 |     91.52 | 
|     LA-SiLU |     **91.97** |     **91.75** | 

#### ResNet32
| Activation | Top1 Best | Top1 Mean | 
| ----------- | --------- | --------- | 
| LA-HardSiLU |     91.74 |     91.47 | 
|        ReLU |     91.96 |     91.79 | 
|        SiLU |     92.16 |     91.96 | 
|     LA-SiLU |     **92.64** |     **92.3** | 
| LA-HardSiLU |     92.16 |     91.84 | 

### ResNet44
|        ReLU |     92.59 |     92.38 | 
|        SiLU |     92.31 |     91.94 | 
|     LA-SiLU |     **93.00** |     **92.67** | 
| LA-HardSiLU |     91.93 |     91.68 | 

