#!/bin/bash

# densenet121 ---------------------------------------
python ./template/RunMe.py --dataset-folder ./datasets/ASBESTOS/ --model-name densenet121 --epochs 30 --experiment-name classification --output-folder ./densenet121/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42

python ./template/RunMe.py --dataset-folder ./datasets/ASBESTOS/ --model-name densenet121 --epochs 30 --experiment-name classification --output-folder ./densenet121/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42 --pretrained

# Resnet18 ---------------------------------------
python ./template/RunMe.py --dataset-folder ./datasets/ASBESTOS/ --model-name resnet18 --epochs 30 --experiment-name classification --output-folder ./resnet18/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42

python ./template/RunMe.py --dataset-folder ./datasets/ASBESTOS/ --model-name resnet18 --epochs 30 --experiment-name classification --output-folder ./resnet18/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42 --pretrained

# Resnet50 ---------------------------------------
python ./template/RunMe.py --dataset-folder ./datasets/ASBESTOS/ --model-name resnet50 --epochs 30 --experiment-name classification --output-folder ./resnet50/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42

python ./template/RunMe.py --dataset-folder ./datasets/ASBESTOS/ --model-name resnet50 --epochs 30 --experiment-name classification --output-folder ./resnet50/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42 --pretrained

# VGG19 ---------------------------------------
"""
Had terrible performance both GPU and accuracy wise

python ./template/RunMe.py --dataset-folder ./datasets/ASBESTOS/ --model-name vgg19 --epochs 30 --experiment-name classification --output-folder ./output/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5

python ./template/RunMe.py --dataset-folder ./datasets/ASBESTOS/ --model-name vgg19 --epochs 30 --experiment-name classification --output-folder ./output/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --pretrained


# VGG19_bn ---------------------------------------
python ./template/RunMe.py --dataset-folder ./datasets/ASBESTOS/ --model-name vgg19_bn --epochs 30 --experiment-name classification --output-folder ./output/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5

python ./template/RunMe.py --dataset-folder ./datasets/ASBESTOS/ --model-name vgg19_bn --epochs 30 --experiment-name classification --output-folder ./output/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --pretrained

"""