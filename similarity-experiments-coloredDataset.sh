#!/bin/bash

#Experiments for Colored Dataset

# Resnet18 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/ --model-name resnet18 --epochs 5 --experiment-name similarity --output-folder ../output/Colored/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --ignoregit

python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/ --model-name resnet18 --epochs 5 --experiment-name similarity --output-folder ../output/Colored/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --pretrained --ignoregit

# Densenet121 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/ --model-name densenet121 --epochs 5 --experiment-name similarity --output-folder ../output/similarity/Colored/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --ignoregit

python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/ --model-name densenet121 --epochs 5 --experiment-name similarity --output-folder ../output/Colored/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --pretrained --ignoregit

#InceptionV3
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/ --model-name inception_v3 --epochs 5 --experiment-name similarity --output-folder ../output/Colored/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --ignoregit

python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/ --model-name inception_v3 --epochs 5 --experiment-name similarity --output-folder ../output/Colored/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --pretrained --ignoregit

# VGG19 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/ --model-name vgg19 --epochs 5 --experiment-name similarity --output-folder ../output/Colored/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --ignoregit

python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/ --model-name vgg19 --epochs 5 --experiment-name similarity --output-folder ../output/Colored/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --pretrained --ignoregit

# VGG19_bn ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/ --model-name vgg19_bn --epochs 5 --experiment-name similarity --output-folder ../output/Colored/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --ignoregit

python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/ --model-name vgg19_bn --epochs 5 --experiment-name similarity --output-folder ../output/Colored/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --pretrained --ignoregit

# VGG11 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/ --model-name vgg11 --epochs 5 --experiment-name similarity --output-folder ../output/Colored/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --ignoregit

python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/ --model-name vgg11 --epochs 5 --experiment-name similarity --output-folder ../output/Colored/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --pretrained --ignoregit

# VGG11_bn ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/ --model-name vgg11_bn --epochs 5 --experiment-name similarity --output-folder ../output/Colored/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --ignoregit

python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/ --model-name vgg11_bn --epochs 5 --experiment-name similarity --output-folder ../output/Colored/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --pretrained --ignoregit

# Densenet201 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/ --model-name densenet201 --epochs 5 --experiment-name similarity --output-folder ../output/Colored/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --ignoregit

python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/ --model-name densenet201 --epochs 5 --experiment-name similarity --output-folder ../output/Colored/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --pretrained --ignoregit

# Resnet153 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/ --model-name resnet153 --epochs 5 --experiment-name similarity --output-folder ../output/Colored/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --ignoregit

python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi/ColoredDataset/ --model-name resnet153 --epochs 5 --experiment-name similarity --output-folder ../output/Colored/ --lr 0.01 --momentum 0.9 --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --pretrained --ignoregit


