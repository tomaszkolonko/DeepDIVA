#!/usr/bin/env bash

# Resnet50 ---------------------------------------
python ./template/RunMe.py --dataset-folder ./datasets/ASBESTOS/ --model-name resnet50 --epochs 30 --experiment-name classification --output-folder ./resnet50/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42

python ./template/RunMe.py --dataset-folder ./datasets/ASBESTOS/ --model-name resnet50 --epochs 30 --experiment-name classification --output-folder ./resnet50/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42 --pretrained
