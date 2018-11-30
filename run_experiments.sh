#!/bin/bash

# densenet121 ---------------------------------------
python ./template/RunMe.py --dataset-folder ../ASBESTOS_A_TRAIN_VAL/ --output-folder ../output/ --model-name densenet121 --epochs 30 --experiment-name classification --output-folder ./densenet121/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42

python ./template/RunMe.py --dataset-folder ../ASBESTOS_A_TRAIN_VAL/ --output-folder ../output/ --model-name densenet121 --epochs 30 --experiment-name classification --output-folder ./densenet121/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42 --pretrained

# Resnet18 ---------------------------------------
python ./template/RunMe.py --dataset-folder ../ASBESTOS_A_TRAIN_VAL/ --output-folder ../output/ --model-name resnet18 --epochs 30 --experiment-name classification --output-folder ./resnet18/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42

python ./template/RunMe.py --dataset-folder ../ASBESTOS_A_TRAIN_VAL/ --output-folder ../output/ --model-name resnet18 --epochs 30 --experiment-name classification --output-folder ./resnet18/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42 --pretrained

# Resnet50 ---------------------------------------
python ./template/RunMe.py --dataset-folder ../ASBESTOS_A_TRAIN_VAL/ --output-folder ../output/ --model-name resnet50 --epochs 30 --experiment-name classification --output-folder ./resnet50/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42

python ./template/RunMe.py --dataset-folder ../ASBESTOS_A_TRAIN_VAL/ --output-folder ../output/ --model-name resnet50 --epochs 30 --experiment-name classification --output-folder ./resnet50/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42 --pretrained
