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
| Data    | Model    |  Activation | Top1 Best | Top1 Mean | 
| ------- | -------- | ----------- | --------- | --------- | 
| CIAR10  | ResNet20 |        ReLU |     91.68 |     91.42 | 
| CIAR10  | ResNet20 |        SiLU |     91.92 |     91.52 | 
| CIAR10  | ResNet20 |     LA-SiLU |     **91.97** |     **91.75** | 
| CIAR10  | ResNet20 | LA-HardSiLU |     91.74 |     91.47 | 
| ------- | -------- | ----------- | --------- | --------- | 
| CIAR10  | ResNet32 |        ReLU |     91.96 |     91.79 | 
| CIAR10  | ResNet32 |        SiLU |     92.16 |     91.96 | 
| CIAR10  | ResNet32 |     LA-SiLU |     **92.64** |     **92.3** | 
| CIAR10  | ResNet32 | LA-HardSiLU |     92.16 |     91.84 | 
| ------- | -------- | ----------- | --------- | --------- | 
| CIAR10  | ResNet44 |        ReLU |     92.59 |     92.38 | 
| CIAR10  | ResNet44 |        SiLU |     92.31 |     91.94 | 
| CIAR10  | ResNet44 |     LA-SiLU |     **93.00** |     **92.67** | 
| CIAR10  | ResNet44 | LA-HardSiLU |     91.93 |     91.68 | 
| ------- | -------- | ----------- | --------- | --------- | 

