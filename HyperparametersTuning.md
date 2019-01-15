# Impact of Architectural Features and Pre-training on the Performance of Deep Neural Networks used in the Analysis of Historical Document

This document provides all the hyperparameters for the different architectures used in the above mentioned paper. All runs have been done with the [DeepDIVA framework](https://github.com/DIVA-DIA/DeepDIVA) We optimized the parameters with [SIGOPT](https://sigopt.com). The learning rate decay divides the learning rate by 10 and happens every N epochs.



## Hyperparameters for Classification (Style Classification)

For the Style Classification task on the [CLaMM dataset](http://clamm.irht.cnrs.fr/wp-content/uploads/ICDAR2017_CLaMM_Training.zip) we optimized the parameters for the following architectures:

|               | Batch Size    | Learning Rate   | lr-decay      | Momentum      | weight_decay   | test accuracy |
|-------------: |-------------: |:---------------:|:-------------:|:-------------:|:--------------:|:-------------:|
| VGG19_bn      | 64            | 0.04509         | 20            | 0.11104       | 0.00680        | 36.97 %       |
| Resnet152     | 64            | 0.06221         | 20            | 0.54338       | 0.00771        | 34.78 %       |
| Inception_v3  | 64            | 0.08231         | 20            | 0.67468       | 0.00689        | 42.72 %       |
| Densenet121   | 64            | 0.08750         | 20            | 0.32541       | 0.00389        | 42.17 %       | 

We let SIGOPT optimize all the hyperparameters again for the runs with pre-training with the following results:

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
| VGG19_bn      | 64            | 0.02622         | 20            | 0.23803       | 0.00869        | 22.66 %       |
| Resnet152     | 64            | 0.07116         | 20            | 0.49010       | 0.01000        | 20.61 %       |
| Inception_v3  | 64            | 0.01523         | 20            | 0.95131       | 0.00674        | 22.36 %       |
| Densenet121   | 64            | 0.04435         | 20            | 0.34480       | 0.00848        | 27.26 %       |

We let SIGOPT optimize all the hyperparameters again for the runs with pre-training with the following results:

|               | Batch Size    | Learning Rate   | lr-decay      | Momentum      | weight_decay   | test accuracy |
|-------------: |:-------------:|:---------------:|:-------------:|:-------------:|:--------------:|:-------------:|
| VGG19_bn      | 64            | 0.05288         | 20            | 0.49782       | 0.00001        | 32.12 %       |
| Resnet152     | 64            | xxxxxxx         | 20            | xxxxxxx       | xxxxxxx        | xxxxxxx %       |
| Inception_v3  | 64            | 0.01788         | 20            | 0.81922       | 0.00621        | 31.92 %       |
| Densenet121   | 64            | xxxxxxx         | 20            | xxxxxxx       | xxxxxxx        | xxxxxxx %       |

## Hyperparameters for Similarity

## Hyperparameters for Segmentation



