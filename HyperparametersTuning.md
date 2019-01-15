# Impact of Architectural Features and ImageNet pre-training on the Performance of Deep Neural Networks used in the Analysis of Historical Document

This document provides all the hyperparameters for the different architectures used in the above mentioned paper. All runs have been done with the [DeepDIVA framework](https://github.com/DIVA-DIA/DeepDIVA) We optimized the parameters with [SIGOPT](https://sigopt.com). The learning rate decay divides the learning rate by 10 and happens every N epochs.



## Hyperparameters for Classification (Style Classification)

For the Style Classification task on the [CLaMM dataset](http://clamm.irht.cnrs.fr/wp-content/uploads/ICDAR2017_CLaMM_Training.zip) we optimized the parameters for the following architectures:

|               | Batch Size    | Learning Rate   | lr-decay      | Momentum      | weight_decay   | test accuracy |
|-------------: |-------------: |:---------------:|:-------------:|:-------------:|:--------------:|:-------------:|
| VGG19_bn      | 64            | 0.04509         | 20            | 0.11104       | 0.00680        | 36.97 %       |
| Resnet152     | 64            | 0.06221         | 20            | 0.54338       | 0.00771        | 34.78 %       |
| Inception_v3  | 64            | 0.08231         | 20            | 0.67468       | 0.00689        | 42.72 %       |
| Densenet121   | 64            | 0.08750         | 20            | 0.32541       | 0.00389        | 42.17 %       | 

We let SIGOPT optimize all the hyperparameters again for the runs with ImageNet pre-training with the following results:

|               | Batch Size    | Learning Rate   | lr-decay      | Momentum      | weight_decay   | test accuracy |
|-------------: |:-------------:|:---------------:|:-------------:|:-------------:|:--------------:|:-------------:|
| VGG19_bn      | 64            | 0.09656         | 20            | 0.65385       | 0.00435        | 47.27 %       |
| Resnet152     | 64            | 0.02609         | 20            | 0.74250       | 0.00895        | 44.42 %       |
| Inception_v3  | 64            | 0.08782         | 20            | 0.25135       | 0.00956        | 48.82 %       |
| Densenet121   | 64            | 0.05002         | 20            | 0.42603       | 0.00575        | 45.92 %       | 

## Hyperparameters for Classification (Manuscript Dating)

For the Manuscript Dating Classification task on the [CLaMM dataset](http://clamm.irht.cnrs.fr/wp-content/uploads/ICDAR2017_CLaMM_Training.zip) we optimized the parameters for the following architectures:

|               | Batch Size    | Learning Rate   | lr-decay      | Momentum      | weight_decay   | test accuracy |
|-------------: |:-------------:|:---------------:|:-------------:|:-------------:|:--------------:|:-------------:|
| VGG19_bn      | 64            | xxxxxxx         | 20            | xxxxxxx       | xxxxxxx        | xxxxxxx %       |
| Resnet152     | 64            | xxxxxxx         | 20            | xxxxxxx       | xxxxxxx        | xxxxxxx %       |
| Inception_v3  | 64            | xxxxxxx         | 20            | xxxxxxx       | xxxxxxx        | xxxxxxx %       |
| Densenet121   | 64            | xxxxxxx         | 20            | xxxxxxx       | xxxxxxx        | xxxxxxx %       |

We let SIGOPT optimize all the hyperparameters again for the runs with ImageNet pre-training with the following results:

|               | Batch Size    | Learning Rate   | lr-decay      | Momentum      | weight_decay   | test accuracy |
|-------------: |:-------------:|:---------------:|:-------------:|:-------------:|:--------------:|:-------------:|
| VGG19_bn      | 64            | xxxxxxx         | 20            | xxxxxxx       | xxxxxxx        | xxxxxxx %       |
| Resnet152     | 64            | xxxxxxx         | 20            | xxxxxxx       | xxxxxxx        | xxxxxxx %       |
| Inception_v3  | 64            | xxxxxxx         | 20            | xxxxxxx       | xxxxxxx        | xxxxxxx %       |
| Densenet121   | 64            | xxxxxxx         | 20            | xxxxxxx       | xxxxxxx        | xxxxxxx %       |

## Hyperparameters for Similarity

For the Historical Writer Identification task on the [Historical-WI dataset](https://scriptnet.iit.demokritos.gr/competitions/6/) we optimized the parameters for the following architectures:

|               | Batch Size      | Learning Rate   | Learning Rate Decay | Momentum      | Weight Decay   | Output Channels | Test mAP      |
|-------------: |:---------------:|:---------------:|:-------------------:|:--------------:|:-------------:|:---------------:|:-------------:|
| VGG19_bn      | 32              | 0.01998         | n/a                 | 0.15635       | 0.00785        | 128             | 2.645 %       | 
| Resnet152     | 32              | 0.00817         | n/a                 | 0.40406       | 0.00565        | 128             | 6.570 %       |
| Inception_v3  | 32              | 0.09931         | n/a                 | 0.31808       | 0.00976        | 128             | 17.49 %       |
| Densenet121   | 32              | 0.03354         | n/a                 | 0.21808       | 0.00231        | 128             | 13.97 %       |

We let SIGOPT optimize all the hyperparameters again for the runs with ImageNet pre-training with the following results:

|               | Batch Size      | Learning Rate   | Learning<br/>Rate<br/>Decay | Momentum      | Weight Decay   | Output Channels | Test mAP      |
|-------------: |:---------------:|:---------------:|:---------------------------:|:-------------:|:--------------:|:---------------:|:-------------:|
| VGG19_bn      | 32              | 0.01554         | n/a                         | 0.48831       | 0.00959        | 128             | 7.648 %       | 
| Resnet152     | 32              | 0.01366         | n/a                         | 0.36760       | 0.00900        | 128             | 13.71 %       |
| Inception_v3  | 32              | 0.03608         | n/a                         | 0.31797       | 0.00107        | 128             | 18.67 %       |
| Densenet121   | 32              | XX         | n/a                 | 0.33007       | 0.00588        | 128             | 19.21 %       |


## Hyperparameters for Segmentation

For the Competition on Layout Analysis for Challenging Medieval Manuscripts task on the [DIVA-HisDB dataset](https://diuf.unifr.ch/main/hisdoc/icdar2017-hisdoc-layout-comp) we optimized the parameters for the following architectures:

|               | Batch Size | Learning Rate   | Learning<br/>Rate<br/>Decay | Momentum      | Weight Decay   | Crop Size | Crops per Page | Pages in Memory | Test meanIU   |
|-------------: |:----------:|:---------------:|:---------------------------:|:-------------:|:--------------:|:---------:|:--------------:|:---------------:|:-------------:|
| UNet          | 8          | 0.04206         | 15                          | 0.11280       | 0.00594        | 256       | 200            | 3               | 58.56 %       | 
| FC-Densenet57 | 8          | 0.02318         | 15                          | 0.32149       | 0.00549        | 256       | 200            | 3               | 58.36 %       |


