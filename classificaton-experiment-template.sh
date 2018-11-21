#!/bin/bash

path="$1"

# Resnet18 ---------------------------------------
python ./template/RunMe.py --runner-class image_classification --dataset-folder "$path" \
--model-name resnet18 --epochs 50 --experiment-name classification --output-folder ../output/ --ignoregit --lr 0.01 --momentum 0.9 \
--decay-lr 20

python ./template/RunMe.py --runner-class image_classification --dataset-folder "$path" \
 --model-name resnet18 --epochs 50 --experiment-name classification --output-folder ../output/ --ignoregit --lr 0.01 --momentum 0.9 \
 --decay-lr 20 --pretrained

# VGG19 ---------------------------------------
python ./template/RunMe.py --runner-class image_classification --dataset-folder "$path" \
 --model-name vgg19 --epochs 50 --experiment-name classification --output-folder ../output/ --ignoregit --lr 0.01 --momentum 0.9 \
 --decay-lr 20

python ./template/RunMe.py --runner-class image_classification --dataset-folder "$path" \
 --model-name vgg19 --epochs 50 --experiment-name classification --output-folder ../output/ --ignoregit --lr 0.01 --momentum 0.9 \
 --decay-lr 20 --pretrained

# VGG19_bn ---------------------------------------
python ./template/RunMe.py --runner-class image_classification --dataset-folder "$path" \
 --model-name vgg19_bn --epochs 50 --experiment-name classification --output-folder ../output/ --ignoregit --lr 0.01 --momentum 0.9 \
 --decay-lr 20

python ./template/RunMe.py --runner-class image_classification --dataset-folder "$path" \
 --model-name vgg19_bn --epochs 50 --experiment-name classification --output-folder ../output/ --ignoregit --lr 0.01 --momentum 0.9 \
 --decay-lr 20 --pretrained

# Densenet121 ---------------------------------------
python ./template/RunMe.py --runner-class image_classification --dataset-folder "$path" \
 --model-name densenet121 --epochs 50 --experiment-name classification --output-folder ../output/ --ignoregit --lr 0.01 --momentum 0.9 \
 --decay-lr 20

python ./template/RunMe.py --runner-class image_classification --dataset-folder "$path" \
 --model-name 121 --epochs 50 --experiment-name classification --output-folder ../output/ --ignoregit --lr 0.01 --momentum 0.9 \
 --decay-lr 20 --pretrained


