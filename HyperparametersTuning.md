# Impact of Architectural Features and Pre-training on the Performance of Deep Neural Networks used in the Analysis of Historical Document

This document provides all the hyperparameters for the different architectures used in the above mentioned paper. All runs have been done with the [DeepDIVA framework](https://github.com/DIVA-DIA/DeepDIVA). The learning rate decay divides the learning rate by 10 and happens every N epochs.



## Hyperparameters for Style Classification

For the Style Classification task on the [CLaMM dataset](http://clamm.irht.cnrs.fr/wp-content/uploads/ICDAR2017_CLaMM_Training.zip) we optimized the parameters the parameters with [SIGOPT](https://sigopt.com) for the following architectures:

|               | Batch Size    | Learning Rate   | Learning<br/>Rate<br/>Decay  | Momentum      | Weight Decay   | Test Accuracy |
|-------------: |-------------: |:---------------:|:----------------------------:|:-------------:|:--------------:|:-------------:|
| CNN_basic     | 64            | 0.01000         | 20                           | 0.9           | 0              | 7.10 %        |
| VGG19_bn      | 64            | 0.04509         | 20                           | 0.11104       | 0.00680        | 36.97 %       |
| Resnet152     | 64            | 0.06221         | 20                           | 0.54338       | 0.00771        | 34.78 %       |
| Inception_v3  | 64            | 0.08231         | 20                           | 0.67468       | 0.00689        | 42.72 %       |
| Densenet121   | 64            | 0.08750         | 20                           | 0.32541       | 0.00389        | 42.17 %       | 

We let SIGOPT optimize all the hyperparameters again for the runs with ImageNet pre-training with the following results:

|               | Batch Size    | Learning Rate   | Learning<br/>Rate<br/>Decay | Momentum      | Weight Decay   | Test Accuracy |
|-------------: |:-------------:|:---------------:|:---------------------------:|:-------------:|:--------------:|:-------------:|
| CNN_basic     | -             | -               | -                           | -             | -              | -             |
| VGG19_bn      | 64            | 0.09656         | 20                          | 0.65385       | 0.00435        | 47.27 %       |
| Resnet152     | 64            | 0.02609         | 20                          | 0.74250       | 0.00895        | 44.42 %       |
| Inception_v3  | 64            | 0.08782         | 20                          | 0.25135       | 0.00956        | 48.82 %       |
| Densenet121   | 64            | 0.05002         | 20                          | 0.42603       | 0.00575        | 45.92 %       | 

## Hyperparameters for Manuscript Dating

For the Manuscript Dating Classification task on the [CLaMM dataset](http://clamm.irht.cnrs.fr/wp-content/uploads/ICDAR2017_CLaMM_Training.zip) we optimized the parameters the parameters with [SIGOPT](https://sigopt.com) for the following architectures:

|               | Batch Size    | Learning Rate   | Learning<br/>Rate<br/>Decay | Momentum      | Weight Decay   | Test Accuracy |
|-------------: |:-------------:|:---------------:|:---------------------------:|:-------------:|:--------------:|:-------------:|
| CNN_basic     | 64            | 0.01000         | 20                          | 0.9           | 0              | 11.21 %       |
| VGG19_bn      | 64            | 0.02622         | 20                          | 0.23803       | 0.00869        | 22.66 %       |
| Resnet152     | 64            | 0.07116         | 20                          | 0.49010       | 0.01000        | 20.61 %       |
| Inception_v3  | 64            | 0.01523         | 20                          | 0.95131       | 0.00674        | 22.36 %       |
| Densenet121   | 64            | 0.04435         | 20                          | 0.34480       | 0.00848        | 27.26 %       |

We let SIGOPT optimize all the hyperparameters again for the runs with pre-training with the following results:

|               | Batch Size    | Learning Rate   | Learning<br/>Rate<br/>Decay | Momentum      | Weight Decay   | Test Accuracy |
|-------------: |:-------------:|:---------------:|:---------------------------:|:-------------:|:--------------:|:-------------:|
| CNN_basic     | -             | -               | -                           | -             | -              | -             |
| VGG19_bn      | 64            | 0.05288         | 20                          | 0.49782       | 0.00001        | 32.12 %       |
| Resnet152     | 64            | 0.01949         | 20                          | 0.56291       | 0.00498        | 32.82 %       |
| Inception_v3  | 64            | 0.01788         | 20                          | 0.81922       | 0.00621        | 31.92 %       |
| Densenet121   | 64            | 0.03187         | 20                          | 0.20721       | 0.00006        | 31.27 %       |

## Hyperparameters for KMNIST

For the Manuscript Dating Classification task on the [KMNIST dataset](https://github.com/rois-codh/kmnist) the same hyperparameters
were used for all the experiments:

|               | Batch Size    | Learning Rate   | Learning<br/>Rate<br/>Decay | Momentum      | Weight Decay   | Test Accuracy |
|-------------: |:-------------:|:---------------:|:---------------------------:|:-------------:|:--------------:|:-------------:|
| VGG19_bn      | 64            | 0.01            | 30                          | 0.09          | 0              |  %       |
| Resnet152     | 64            | 0.01            | 30                          | 0.09          | 0              |  %       |
| Inception_v3  | 64            | 0.01            | 30                          | 0.09          | 0              |  %       |
| Densenet121   | 64            | 0.01            | 30                          | 0.09          | 0              |  %       |


## Hyperparameters for Content-Based Image Retrieval

For the Historical Writer Identification task on the [Historical-WI dataset](https://scriptnet.iit.demokritos.gr/competitions/6/) we optimized the parameters the parameters with [SIGOPT](https://sigopt.com) for the following architectures:

|               | Batch Size      | Learning Rate   | Learning<br/>Rate<br/>Decay | Momentum       | Weight Decay  | Output Channels | Test mAP      |
|-------------: |:---------------:|:---------------:|:---------------------------:|:--------------:|:-------------:|:---------------:|:-------------:|
| 3-layer CNN   | 32              | 0.001           | n/a                         | 0              | 0             | 128             | 11.4 %       |
| VGG19_bn      | 32              | 0.01998         | n/a                         | 0.15635        | 0.00785       | 128             | 14.6 %       | 
| Resnet152     | 32              | 0.00817         | n/a                         | 0.40406        | 0.00565       | 128             | 24.7 %       |
| Inception_v3  | 32              | 0.007           | n/a                         | 0.31808        | 0.00976       | 128             | 9.1  %       |
| Densenet121   | 32              | 0.03354         | n/a                         | 0.21808        | 0.00231       | 128             | 27.2 %       |

We let SIGOPT optimize all the hyperparameters again for the runs with ImageNet pre-training with the following results:

|               | Batch Size      | Learning Rate   | Learning<br/>Rate<br/>Decay | Momentum      | Weight Decay   | Output Channels | Test mAP      |
|-------------: |:---------------:|:---------------:|:---------------------------:|:-------------:|:--------------:|:---------------:|:-------------:|
| VGG19_bn      | 32              | 0.001554        | n/a                         | 0.48831       | 0.00959        | 128             | xx %       | 
| Resnet152     | 32              | 0.01366         | n/a                         | 0.36760       | 0.00900        | 128             | 22.1 %       |
| Inception_v3  | 32              | 0.03608         | n/a                         | 0.31797       | 0.00107        | 128             | 26.1 %       |
| Densenet121   | 32              | 0.01662         | n/a                         | 0.17825       | 0.00254        | 128             | 34.6 %       | 


## Hyperparameters for Segmentation

Hyperparemters for the architectures used for the Competition on Layout Analysis for Challenging Medieval Manuscripts task of the [DIVA-HisDB dataset](https://diuf.unifr.ch/main/hisdoc/icdar2017-hisdoc-layout-comp) 
(evaluation is performed using [this layout analysis tool](https://github.com/DIVA-DIA/DIVA_Layout_Analysis_Evaluator)): 

|               | Batch Size | Learning Rate   | Learning<br/>Rate<br/>Decay | Momentum  | Weight Decay   | Crop Size | Crops per Page | Pages in Memory | Test meanIU CB55 | Test meanIU CS863 | Test meanIU CS18 |
|-------------: |:----------:|:---------------:|:---------------------------:|:---------:|:--------------:|:---------:|:--------------:|:---------------:|:----------------:|:-----------------:|:----------------:|
| UNet          | 32         | 0.005           | 24                          | 0.9       | 0              | 256       | 1000           | 3               | 92.10 %          | XX.XX %           | XX.XX %          | 
| FC-Densenet57 | 8          | 0.005           | 24                          | 0.9       | 0              | 256       | 1000           | 3               | XX.XX %          | XX.XX %           | XX.XX %          |


