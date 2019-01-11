#!/bin/bash

#Experiments for Colored Dataset NOT PRETRAINED

# Densenet121 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ --model-name densenet121 --epochs 25 --experiment-name similarity-longrun --output-folder ../output/ --decay-lr 25 --n-triplets 1000000 --output-channels 128 --map auto --ignoregit --batch-size 32 --regenerate-every 3 -j 16 --lr 0.035 --momentum 0.22 --weight-decay 0.0023

# VGG19_bn ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col/ --model-name vgg19_bn --epochs 25 --experiment-name similarity-longrun --output-folder ../output/ --decay-lr 25 --n-triplets 1000000 --output-channels 128 --map auto --ignoregit --batch-size 32 --regenerate-every 3 -j 16 --lr 0.2 --momentum 0.15 --weight-decay 0.0078

#InceptionV3
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ --model-name inception_v3 --epochs 25 --experiment-name similarity-longrun --output-folder ../output/ --decay-lr 25 --n-triplets 1000000 --output-channels 128 --map auto --ignoregit --batch-size 32 --regenerate-every 3 -j 16 --lr 0.1 --momentum 0.32 --weight-decay 0.0098

# Resnet152 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ --model-name resnet152 --epochs 25 --experiment-name similarity-longrun --output-folder ../output/ --decay-lr 25 --n-triplets 1000000 --output-channels 128 --map auto --ignoregit --batch-size 32 --regenerate-every 3 -j 16 --lr 0.0081 --momentum 0.4 --weight-decay 0.0056


exit 1
#*********************************************************************************************
#Experiments for Colored Dataset PRETRAINED

#InceptionV3
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ --model-name inception_v3 --epochs 25 --experiment-name similarity-longrun-pretrained --output-folder ../output/ --decay-lr 20 --n-triplets 1000000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 3 -j 16 --lr --momentum --weight-decay --batch-size 32

# VGG19_bn ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ --model-name vgg19_bn --epochs 25 --experiment-name similarity-longrun-pretrained --output-folder ../output/ --decay-lr 20 --n-triplets 1000000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 3 -j 16 --lr --momentum --weight-decay --batch-size 32

# Resnet152 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ --model-name resnet152 --epochs 25 --experiment-name similarity-longrun-pretrained --output-folder ../output/ --decay-lr 20 --n-triplets 1000000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 3 -j 16 --lr --momentum --weight-decay --batch-size 32

# Densenet121 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ --model-name densenet121 --epochs 25 --experiment-name similarity-longrun-pretrained --output-folder ../output/ --decay-lr 20 --n-triplets 1000000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 3 -j 16 --lr --momentum --weight-decay --batch-size 32
