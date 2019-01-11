# Impact of Architectural Features and Pre-training on the Performance of Deep Neural Networks used in the Analysis of Historical Document

This document provides all the hyperparameters for the different architectures used in the above mentioned paper. All runs have been done with the [DeepDIVA framework](https://github.com/DIVA-DIA/DeepDIVA) We optimized the parameters with [SIGOPT](https://sigopt.com)



## Hyperparameters for Classification (Style Classification)

For the Style Classification task on the [CLaMM dataset]() we optimized the parameters for the following architectures:

|               | Learning Rate   | Momentum      | weight_decay   | test accuracy |
|-------------: |:---------------:|:-------------:|:--------------:|:-------------:|
| VGG19_bn      | 0.04509         | 0.11104       | 0.00680        | 36.97 %       |
| Resnet152     | 0.06221         | 0.54338       | 0.00771        | 34.78 %       |
| Inception_v3  | 0.08231         | 0.67468       | 0.00689        | 42.72 %       |
| Densenet121   | 0.08750         | 0.32541       | 0.00389        | 42.17 %       | 

We let SIGOPT optimize all the hyperparameters again for the runs with pre-training with the following results:

|               | Learning Rate   | Momentum      | weight_decay   | test accuracy |
|-------------: |:---------------:|:-------------:|:--------------:|:-------------:|
| VGG19_bn      | 0.09656         | 0.65385       | 0.00435        | 47.27 %       |
| Resnet152     | 0.02609         | 0.74250       | 0.00895        | 44.42 %       |
| Inception_v3  | 0.08782         | 0.25135       | 0.00956        | 48.82 %       |
| Densenet121   | 0.05002         | 0.42603       | 0.00575        | 45.92 %       | 

## Hyperparameters for Classification (Manuscript Dating)

For the Manuscript Dating Classification task on the [CLaMM dataset]() we optimized the parameters for the following architectures:

|               | Learning Rate   | Momentum      | weight_decay   | test accuracy |
|-------------: |:---------------:|:-------------:|:--------------:|:-------------:|
| VGG19_bn      | xxxxxxx         | xxxxxxx       | xxxxxxx        | xxxxxxx %       |
| Resnet152     | xxxxxxx         | xxxxxxx       | xxxxxxx        | xxxxxxx %       |
| Inception_v3  | xxxxxxx         | xxxxxxx       | xxxxxxx        | xxxxxxx %       |
| Densenet121   | xxxxxxx         | xxxxxxx       | xxxxxxx        | xxxxxxx %       |

We let SIGOPT optimize all the hyperparameters again for the runs with pre-training with the following results:

|               | Learning Rate   | Momentum      | weight_decay   | test accuracy |
|-------------: |:---------------:|:-------------:|:--------------:|:-------------:|
| VGG19_bn      | xxxxxxx         | xxxxxxx       | xxxxxxx        | xxxxxxx %       |
| Resnet152     | xxxxxxx         | xxxxxxx       | xxxxxxx        | xxxxxxx %       |
| Inception_v3  | xxxxxxx         | xxxxxxx       | xxxxxxx        | xxxxxxx %       |
| Densenet121   | xxxxxxx         | xxxxxxx       | xxxxxxx        | xxxxxxx %       |

## Hyperparameters for Similarity

## Hyperparameters for Segmentation



