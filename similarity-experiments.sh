#!/bin/bash

#Experiments for Colored Dataset NOT PRETRAINED with optimized parameters

# Densenet121 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder /data/ltp/historical_wi/ColoredDataset/ --model-name densenet121 --epochs 10 --experiment-name similarity-opt-multicrop --output-folder ../output/ --decay-lr 12 --n-triplets 1500000 --output-channels 128 --map auto --ignoregit --batch-size 32 --regenerate-every 1 --lr 0.03354 --momentum 0.21808 --weight-decay 0.00231 --seed 42 -j 16 --gpu-id 0

# VGG19_bn ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder /data/ltp/historical_wi/ColoredDataset/ --model-name vgg19_bn --epochs 10 --experiment-name similarity-opt-multicrop --output-folder ../output/ --decay-lr 12 --n-triplets 1500000 --output-channels 128 --map auto --ignoregit --batch-size 32 --regenerate-every 1 --lr 0.01998 --momentum 0.15635 --weight-decay 0.00785 --seed 1037 -j 16 --gpu-id 0

#InceptionV3
python ./template/RunMe.py --runner-class triplet --dataset-folder /data/ltp/historical_wi/ColoredDataset/ --model-name inception_v3 --epochs 10 --experiment-name similarity-opt-multicrop --output-folder ../output/ --decay-lr 12 --n-triplets 1500000 --output-channels 128 --map auto --ignoregit --batch-size 32 --regenerate-every 1 --lr 0.007 --momentum 0.31808 --weight-decay 0.00976 --seed 1037 -j 16 --gpu-id 0

# Resnet152 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder /data/ltp/historical_wi/ColoredDataset/ --model-name resnet152 --epochs 10 --experiment-name similarity-opt-multicrop --output-folder ../output/ --decay-lr 12 --n-triplets 1500000 --output-channels 128 --map auto --ignoregit --batch-size 32 --regenerate-every 1 --lr 0.00817 --momentum 0.40406 --weight-decay 0.00565 --seed 42 -j 16 --gpu-id 0

# CNN basic ---------------------------------------
python ./template/RunMe.py  --runner-class triplet --dataset-folder /data/ltp/historical_wi/ColoredDataset/ --model-name CNN_basic --epochs 10 --similarity-old-setting --output-folder ../output/ --n-triplets 1500000 --output-channels 128 --map auto --batch-size 32 --regenerate-every 1 --seed 42 -j --gpu-id 0

#*********************************************************************************************
#Experiments for Colored Dataset PRETRAINED with optimized parameters
# Densenet121 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder /data/ltp/historical_wi/ColoredDataset/ --model-name densenet121 --epochs 10 --experiment-name similarity-opt-multicrop-pretrained --output-folder ../output/ --decay-lr 12 --n-triplets 1500000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 1 --lr 0.03608 --momentum 0.31797 --weight-decay 0.00107 --seed 42 --batch-size 32  -j 16 --gpu-id 0

# VGG19_bn ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder /data/ltp/historical_wi/ColoredDataset/ --model-name vgg19_bn --epochs 10 --experiment-name similarity-opt-multicrop-pretrained --output-folder ../output/ --n-triplets 1500000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 1 --lr 0.001554 --momentum 0.48831 --weight-decay 0.00959 --seed 42 --batch-size 32  -j 16 --gpu-id 0

#InceptionV3
python ./template/RunMe.py --runner-class triplet --dataset-folder /data/ltp/historical_wi/ColoredDataset/ --model-name inception_v3 --epochs 10 --experiment-name similarity-opt-multicrop-pretrained --output-folder ../output/ --decay-lr 12 --n-triplets 1500000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 1 --lr 0.03608 --momentum 0.31797 --weight-decay 0.00107 --seed 1037 --batch-size 32  -j 16 --gpu-id 0

# Resnet152 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder /data/ltp/historical_wi/ColoredDataset/ --model-name resnet152 --epochs 10 --experiment-name similarity-opt-multicrop-pretrained --output-folder ../output/ --decay-lr 12 --n-triplets 1500000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 1 --lr 0.01366 --momentum 0.36760 --weight-decay 0.00900 --seed 42 --batch-size 32  -j 16 --gpu-id 0
