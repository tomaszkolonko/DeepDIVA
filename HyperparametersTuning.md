# Impact of Architectural Features and Pre-training on the Performance of Deep Neural Networks used in the Analysis of Historical Document

This document provides all the hyper-parameters for the different architectures used in the above mentioned paper. All runs have been done with the [DeepDIVA framework](https://github.com/DIVA-DIA/DeepDIVA). The learning rate decay divides the learning rate by 10 and happens every N epochs.

## Hyper-parameters for Character Recognition (Classification)
hyper-parameters for the character recognition task on the [KMNIST dataset](https://github.com/rois-codh/kmnist). 
All architectures (VGG19 BN, Inception V3, ResNet152, DenseNet121) have been trained with the same hyper-parameters for both experiments (from scratch and pre-trained).

| Batch Size    | Learning Rate   | Learning<br/>Rate<br/>Decay | Momentum | Weight Decay   |
|-------------: |:---------------:|:---------------------------:|:--------:|:--------------:|
| 64            | 0.01            | 30                          | 0.9      | 0              |

## Hyper-parameters for Style Classification (Classification)
For the Style Classification task on the [CLaMM dataset](http://clamm.irht.cnrs.fr/wp-content/uploads/ICDAR2017_CLaMM_Training.zip) we optimized the parameters the parameters with [SIGOPT](https://sigopt.com) for the following architectures:

|               | Batch Size    | Learning Rate   | Learning<br/>Rate<br/>Decay  | Momentum      | Weight Decay   |
|-------------: |-------------: |:---------------:|:----------------------------:|:-------------:|:--------------:|
| CNN_basic     | 64            | 0.01000         | 20                           | 0.9           | 0              |
| VGG19_bn      | 64            | 0.04509         | 20                           | 0.11104       | 0.00680        |
| Resnet152     | 64            | 0.06221         | 20                           | 0.54338       | 0.00771        |
| Inception_v3  | 64            | 0.08231         | 20                           | 0.67468       | 0.00689        |
| Densenet121   | 64            | 0.08750         | 20                           | 0.32541       | 0.00389        | 

We let SIGOPT optimize all the hyper-parameters again for the runs with ImageNet pre-training with the following results:

|               | Batch Size    | Learning Rate   | Learning<br/>Rate<br/>Decay | Momentum      | Weight Decay   |
|:-------------:|:---------------:|:---------------------------:|:-------------:|:--------------:|:-------------:|
| VGG19_bn      | 64            | 0.09656         | 20                          | 0.65385       | 0.00435        |
| Resnet152     | 64            | 0.02609         | 20                          | 0.74250       | 0.00895        |
| Inception_v3  | 64            | 0.08782         | 20                          | 0.25135       | 0.00956        |
| Densenet121   | 64            | 0.05002         | 20                          | 0.42603       | 0.00575        | 


## Hyper-parameters for Manuscript Dating (Classification)

For the Manuscript Dating Classification task on the [CLaMM dataset](http://clamm.irht.cnrs.fr/wp-content/uploads/ICDAR2017_CLaMM_Training.zip) we optimized the parameters with [SIGOPT](https://sigopt.com) for the following architectures:

|               | Batch Size    | Learning Rate   | Learning<br/>Rate<br/>Decay | Momentum      | Weight Decay   |
|:-------------:|:---------------:|:---------------------------:|:-------------:|:--------------:|:-------------:|
| CNN_basic     | 64            | 0.01000         | 20                          | 0.9           | 0              |
| VGG19_bn      | 64            | 0.02622         | 20                          | 0.23803       | 0.00869        |
| Resnet152     | 64            | 0.07116         | 20                          | 0.49010       | 0.01000        |
| Inception_v3  | 64            | 0.01523         | 20                          | 0.95131       | 0.00674        |
| Densenet121   | 64            | 0.04435         | 20                          | 0.34480       | 0.00848        |

We let SIGOPT optimize all the hyper-parameters again for the runs with pre-training with the following results:

|               | Batch Size    | Learning Rate   | Learning<br/>Rate<br/>Decay | Momentum      | Weight Decay   | |:-------------:|:---------------:|:---------------------------:|:-------------:|:--------------:|:-------------:|
| VGG19_bn      | 64            | 0.05288         | 20                          | 0.49782       | 0.00001        |
| Resnet152     | 64            | 0.01949         | 20                          | 0.56291       | 0.00498        |
| Inception_v3  | 64            | 0.01788         | 20                          | 0.81922       | 0.00621        |
| Densenet121   | 64            | 0.03187         | 20                          | 0.20721       | 0.00006        |

## Hyper-parameters for Writer Identification (Image Similarity)

For the Historical Writer Identification task on the [Historical-WI dataset](https://scriptnet.iit.demokritos.gr/competitions/6/) we optimized the parameters with [SIGOPT](https://sigopt.com) for the following architectures:

|               | Batch Size      | Learning Rate   | Learning<br/>Rate<br/>Decay | Momentum       | Weight Decay  | Output Channels |
|:---------------:|:---------------:|:---------------------------:|:--------------:|:-------------:|:---------------:|:-------------:|
| CNN_basic     | 32              | 0.001           | n/a                         | 0              | 0             | 128             |
| VGG19_bn      | 32              | 0.01998         | n/a                         | 0.15635        | 0.00785       | 128             | 
| Resnet152     | 32              | 0.00817         | n/a                         | 0.40406        | 0.00565       | 128             |
| Inception_v3  | 32              | 0.007           | n/a                         | 0.31808        | 0.00976       | 128             |
| Densenet121   | 32              | 0.03354         | n/a                         | 0.21808        | 0.00231       | 128             |


We let SIGOPT optimize all the hyper-parameters again for the runs with ImageNet pre-training with the following results:

|               | Batch Size      | Learning Rate   | Learning<br/>Rate<br/>Decay | Momentum      | Weight Decay   | Output Channels |
|:---------------:|:---------------:|:---------------------------:|:-------------:|:--------------:|:---------------:|:-------------:|
| VGG19_bn      | 32              | 0.001554        | n/a                         | 0.48831       | 0.00959        | 128             | 
| Resnet152     | 32              | 0.01366         | n/a                         | 0.36760       | 0.00900        | 128             |
| Inception_v3  | 32              | 0.03608         | n/a                         | 0.31797       | 0.00107        | 128             |
| Densenet121   | 32              | 0.01662         | n/a                         | 0.17825       | 0.00254        | 128             | 


## Hyper-parameters for Segmentation

Hyper-parameters for the architectures used for the Competition on Layout Analysis for Challenging Medieval Manuscripts task of the [DIVA-HisDB dataset](https://diuf.unifr.ch/main/hisdoc/icdar2017-hisdoc-layout-comp).
The same hyper-parameters are used for the ImageNet pre-trained experiments with SegNet and DeepLabV3.

|                              | Batch Size | Learning Rate   | Learning<br/>Rate<br/>Decay | Momentum  | Weight Decay   | Crop Size | # Crops per Training Epoch |
|-----------------------------:|:----------:|:---------------:|:---------------------------:|:---------:|:--------------:|:---------:|:--------------:|
| SegNet (VGG19 BN encoder)    | 32         | 0.001           | 24                          | 0.9       | 0              | 256       | 60'000           | 
| DeepLabV3 (ResNet18 encoder) | 32         | 0.005           | 24                          | 0.9       | 0              | 256       | 60'000           |
| 5-layer CNN (BabyUnet)       | 32         | 0.005           | 24                          | 0.9       | 0              | 256       | 60'000           |
