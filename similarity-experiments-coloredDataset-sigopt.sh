#!/bin/bash

#Experiments for Colored Dataset NOT PRETRAINED

# Densenet121 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small --model-name densenet121 --epochs 30 --experiment-name ls-sigopt-similarity --output-folder ../output/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --ignoregit

#InceptionV3
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small --model-name inception_v3 --epochs 30 --experiment-name ls-sigopt-similarity --output-folder ../output/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --ignoregit

# VGG19_bn ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small --model-name vgg19_bn --epochs 30 --experiment-name ls-sigopt-similarity --output-folder ../output/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --ignoregit

# Resnet152 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small --model-name resnet152 --epochs 30 --experiment-name ls-sigopt-similarity --output-folder ../output/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --ignoregit


exit 1
#*********************************************************************************************
#Experiments for Colored Dataset PRETRAINED

# Densenet121 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small --model-name densenet121 --epochs 10 --experiment-name ls-sigopt-similarity --output-folder ../output/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --pretrained --ignoregit

#InceptionV3
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small --model-name inception_v3 --epochs 10 --experiment-name ls-sigopt-similarity --output-folder ../output/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --pretrained --ignoregit

# VGG19_bn ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small --model-name vgg19_bn --epochs 10 --experiment-name ls-sigopt-similarity --output-folder ../output/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --pretrained --ignoregit

# Resnet152 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small --model-name resnet152 --epochs 10 --experiment-name ls-sigopt-similarity --output-folder ../output/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --pretrained --ignoregit