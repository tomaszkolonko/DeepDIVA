#!/usr/bin/env bash

# Resnet18 ---------------------------------------
python ./template/RunMe.py --dataset-folder ../ASBESTOS_A_TRAIN_VAL/ --output-folder ../output/ --model-name resnet18 --epochs 30 --experiment-name classification --output-folder ./resnet18/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42

python ./template/RunMe.py --dataset-folder ../ASBESTOS_A_TRAIN_VAL/ --output-folder ../output/ --model-name resnet18 --epochs 30 --experiment-name classification --output-folder ./resnet18/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42 --pretrained
