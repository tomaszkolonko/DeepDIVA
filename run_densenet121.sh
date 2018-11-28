#!/usr/bin/env bash

# densenet121 ---------------------------------------
python ./template/RunMe.py --dataset-folder ./datasets/ASBESTOS/ --model-name densenet121 --epochs 30 --experiment-name classification --output-folder ./densenet121/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42

python ./template/RunMe.py --dataset-folder ./datasets/ASBESTOS/ --model-name densenet121 --epochs 30 --experiment-name classification --output-folder ./densenet121/ --ignoregit --lr 0.01 --optimizer-name Adam --decay-lr 5 --seed 42 --pretrained
