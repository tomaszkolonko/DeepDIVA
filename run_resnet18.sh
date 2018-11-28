#!/usr/bin/env bash

# Resnet18 ---------------------------------------
python ./template/RunMe.py --dataset-folder ./datasets/ASBESTOS/ --model-name resnet18 --epochs 30 --experiment-name classification --output-folder ./resnet18/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42

python ./template/RunMe.py --dataset-folder ./datasets/ASBESTOS/ --model-name resnet18 --epochs 30 --experiment-name classification --output-folder ./resnet18/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42 --pretrained
