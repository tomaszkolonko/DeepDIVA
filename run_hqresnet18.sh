#!/usr/bin/env bash

# hq_resnet18 ---------------------------------------
python ./template/RunMe.py --dataset-folder ../ASBESTOS/ --output-folder ../hq_resnet18/ --model-name hqresnet18 --epochs 30 --experiment-name classification --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42

# python ./template/RunMe.py --dataset-folder ../ASBESTOS/ --output-folder ../hq_resnet18/ --model-name hqresnet18 --epochs 30 --experiment-name classification --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42 --pretrained