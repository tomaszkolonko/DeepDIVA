#!/bin/bash

#Experiments for Colored Dataset

# Resnet18 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/
--model-name resnet18 --epochs 50 --experiment-name similarity --output-folder ../output/similarity/Colored/ --ignoregit --lr 0.01 --momentum 0.9
--decay-lr 20 --output-channels 128

python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/
--model-name resnet18 --epochs 50 --experiment-name similarity --output-folder ../output/similarity/Colored/ --ignoregit --lr 0.01 --momentum 0.9
--decay-lr 20 --output-channels 128 --pretrained

# VGG19 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/
--model-name vgg19 --epochs 50 --experiment-name similarity --output-folder ../output/similarity/Colored/ --ignoregit --lr 0.01 --momentum 0.9
--decay-lr 20 --output-channels 128

python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/
--model-name vgg19 --epochs 50 --experiment-name similarity --output-folder ../output/similarity/Colored/ --ignoregit --lr 0.01 --momentum 0.9
--decay-lr 20 --output-channels 128 --pretrained

# VGG19_bn ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/
--model-name vgg19_bn --epochs 50 --experiment-name similarity --output-folder ../output/similarity/Colored/ --ignoregit --lr 0.01 --momentum 0.9
--decay-lr 20 --output-channels 128

python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/
--model-name vgg19_bn --epochs 50 --experiment-name similarity --output-folder ../output/similarity/Colored/ --ignoregit --lr 0.01 --momentum 0.9
--decay-lr 20 --output-channels 128 --pretrained

# Densenet121 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/
--model-name densenet121 --epochs 50 --experiment-name similarity --output-folder ../output/similarity/Colored/ --ignoregit --lr 0.01 --momentum 0.9
--decay-lr 20 --output-channels 128

python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/
--model-name 121 --epochs 50 --experiment-name similarity --output-folder ../output/similarity/Colored/ --ignoregit --lr 0.01 --momentum 0.9
--decay-lr 20 --output-channels 128 --pretrained


