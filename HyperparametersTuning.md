# Impact of Architectural Features and Pre-training on the Performance of Deep Neural Networks used in the Analysis of Historical Document

This document provides all the used hyperparameters for the different architectures used in the above mentioned paper. All runs have been done with the DeepDIVA framework [DeepDIVA framework](https://github.com/DIVA-DIA/DeepDIVA) We optimized the parameters with [SIGOPT](https://sigopt.com)



## Hyperparameters for Classification (Style Classification)

For the Style Classification on the [CLaMM dataset]() we optimized the parameters for the following architectures:

|               | Learning Rate   | Momentum      | weight_decay   | test accuracy |
|-------------: |:---------------:|:-------------:|:--------------:|:-------------:|
| VGG19_bn      | 0.04509         | 0.11104       | 0.00680        | 36.97 %       |
| Resnet152     | 0.06221         | 0.54338       | 0.00771        | 34.78 %       |
| Inception_v3  | 0.08231         | 0.67468       | 0.00689        | 42.72 %       |
| Inception_v3  | 0.08750         | 0.32541       | 0.00389        | 42.17 %       | 

We let SIGOPT optimize all the hyperparameters again for the runs with pre-training with the following results:



## Hyperparameters for Classification (Manuscript Dating)

|               | Learning Rate   | Momentum      | weight_decay   | test accuracy |
|-------------: |:---------------:|:-------------:|:--------------:|:-------------:|
| VGG19_bn      | xxxxxxx         | xxxxxxx       | xxxxxxx        | xxxxxxx %       |
| Resnet152     | xxxxxxx         | xxxxxxx       | xxxxxxx        | xxxxxxx %       |
| Inception_v3  | xxxxxxx         | xxxxxxx       | xxxxxxx        | xxxxxxx %       |
| Inception_v3  | xxxxxxx         | xxxxxxx       | xxxxxxx        | xxxxxxx %       | 

## Hyperparameters for Similarity

## Hyperparameters for Segmentation



