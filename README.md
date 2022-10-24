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
- train/Validation: 55000/5000
- max iteration: around 64000 (until the last epoch end)
- 32X32 random crop
- horizontal flip
- learning rate: 0.1
- optimizer: Momentum (0.9) with weight decay 0.0001
- lr scheduler: 0.1 at iteration 32000 and 48000
- random seed for data split: 0
- random seed for weight initialization: 11*n (n=1,2,...)

## Experiment
- We show the performance of the models with the best and mean accuracy of 5-runs.
- Top-1 accuracy: The model's prediction is correct when the prediction is exactly the same as the true label.
- Top-5 accuracy: The model's prediction is correct when the top-5 scored predictions include the true label.
We show the bias of the activation functions between samples with the standard deviation (std) of loss between the test samples
- std Best : The lowest standard deviation of loss between the test samples among 5-runs.
- std Mean : The mean of standard deviation of loss between the test samples of 5-runs.

### CIFAR10
- CIFAR10 is an image dataset for classification task with 10 classes. 
##### ResNet20
| Activation  | Top1 Best | Top1 Mean | std Best  | std Mean  |
| ----------- | --------- | --------- | --------- | --------- |  
|        ReLU |     91.68 |     91.42 | 1.514     | 1.556     |
|   LeakyReLU |     91.49 |     91.18 | 1.487     | 1.541     |
|        SiLU |     91.92 |     91.52 | 1.564     | 1.596     |
|     LA-SiLU |     **91.97** |     **91.75** | **1.381**     | **1.401**     |
| LA-HardSiLU |     91.74 |     91.47 | 1.417     | 1.447     |
- The result of ResNet20 above is performed with server1, not server3. 

##### ResNet32
| Activation  | Top1 Best | Top1 Mean | std Best  | std Mean  |
| ----------- | --------- | --------- | --------- | --------- |  
|        ReLU |     91.96 |     91.79 | 1.537    | 1.652     |
|   LeakyReLU |     92.23 |     91.87 | 1.582     | 1.647     |
|        SiLU |     92.16 |     91.96 | 1.609    | 1.648     |
|     LA-SiLU |     **92.64** |     **92.3** | **1.433**     | **1.477**     |
| LA-HardSiLU |     92.16 |     91.84 | 1.463     | 1.492     |

##### ResNet44
| Activation  | Top1 Best | Top1 Mean | std Best  | std Mean  |
| ----------- | --------- | --------- | --------- | --------- |  
|        ReLU |     92.59 |     92.38 | 1.536     | 1.641     |
|   LeakyReLU |     92.42 |     92.19 | 1.554     | 1.685     |
|        SiLU |     92.31 |     91.94 | 1.595     | 1.668     |
|     LA-SiLU |     **93.00** |     **92.67** | **1.452**     | **1.478**     |
| LA-HardSiLU |     91.93 |     91.68 | 1.606     | 1.639     |

##### WideResNet28-10 with dropout rate 0.3
| Activation  | Top1 Best | Top1 Mean | std best  | std mean  |
| ----------- | --------- | --------- | --------- | --------- |  
|        ReLU |     94.63 |     94.03 | 1.468     | 1.542     |
|   LeakyReLU |     94.3 |     94.06 | 1.419     | 1.488     |
|        SiLU |     93.6 |     93.16 | 1.374     | 1.542     |
|     LA-SiLU |     **94.87** |     **94.24** | 1.096     | 1.187     |
| LA-HardSiLU |     94.7 |     94.14 | **1.007**     | **1.152**     |

### CIFAR100
- CIFAR100 is an image dataset for classification task with 100 classes. 
##### ResNet20
| Activation  | Top1 Best | Top1 Mean | Top5 Best | Top5 Mean | std best  | std mean  |
| ----------- | --------- | --------- | --------- | --------- | --------- | --------- |
|        ReLU |     66.25 |     65.98 |     89.81 |     89.62 | 2.573     | 2.624     |
|        SiLU |     **66.68** |     66.09 |     90.23 |     89.59 | 2.789     | 2.815     |
|     LA-SiLU |     66.65 |     **66.42** |     **90.32** |     **89.97** | 2.430     | 2.452     |
| LA-HardSiLU |     66.64 |     66.30 |     90.04 |     89.84 | **2.361**     | **2.410**     |

##### ResNet32
| Activation  | Top1 Best | Top1 Mean | Top5 Best | Top5 Mean | std best  | std mean  | 
| ----------- | --------- | --------- | --------- | --------- | --------- | --------- |
|        ReLU |     67.63 |     67.38 |     **90.34** |     90.09 | 3.000     | 3.050     |
|        SiLU |     **68.17** |     67.45 |     90.05 |     89.92 | 3.175     | 3.234     |
|     LA-SiLU |     67.99 |     **67.84** |     90.33 |     **90.11** | 2.752     | 2.783     |
| LA-HardSiLU |     67.88 |     67.16 |     89.96 |     89.73 | **2.707**     | **2.759**     |

##### ResNet44
| Activation  | Top1 Best | Top1 Mean | Top5 Best | Top5 Mean | std best  | std mean  | 
| ----------- | --------- | --------- | --------- | --------- | --------- | --------- |
|        ReLU |     68.68 |     68.44 |     90.47 |     90.29 | 3.184     | 3.237     |
|        SiLU |     **68.99** |     68.37 |     90.55 |     90.04 | 3.316     | 3.344     |
|     LA-SiLU |     68.89 |     **68.68** |     **90.80** |     **90.42** | **2.787**     | **2.815**     |
| LA-HardSiLU |     67.67 |     66.80 |     89.60 |     89.34 | 2.841     | 2.887     |

- More experiments will be updated. 
