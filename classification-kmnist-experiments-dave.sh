#!/bin/bash

# lr = 0.01

# Resnet152 -------------------------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder ../datasets/KMNIST --model-name resnet152 --epochs 100 --experiment-name classification \
    --output-folder ../output --ignoregit --lr 0.01 --momentum 0.9 --decay-lr 30

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder ../datasets/KMNIST --model-name resnet152 --epochs 100 --experiment-name classification \
    --output-folder ../output --ignoregit --lr 0.01 --momentum 0.9 --decay-lr 30 --pretrained

# VGG19 -------------------------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder ../datasets/KMNIST --model-name vgg19_bn --epochs 100 --experiment-name classification \
    --output-folder ../output --ignoregit --lr 0.01 --momentum 0.9 --decay-lr 30

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder ../datasets/KMNIST --model-name vgg19_bn --epochs 100 --experiment-name classification \
    --output-folder ../output --ignoregit --lr 0.01 --momentum 0.9 --decay-lr 30 --pretrained

# densenet121 -------------------------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder ../datasets/KMNIST --model-name densenet121 --epochs 100 --experiment-name classification \
    --output-folder ../output --ignoregit --lr 0.01 --momentum 0.9 --decay-lr 30

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder ../datasets/KMNIST --model-name densenet121 --epochs 100 --experiment-name classification \
    --output-folder ../output --ignoregit --lr 0.01 --momentum 0.9 --decay-lr 30 --pretrained

# inception -------------------------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder ../datasets/KMNIST --model-name inception_v3 --epochs 100 --experiment-name classification \
    --output-folder ../output --ignoregit --lr 0.01 --momentum 0.9 --decay-lr 30

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder ../datasets/KMNIST --model-name inception_v3 --epochs 100 --experiment-name classification \
    --output-folder ../output --ignoregit --lr 0.01 --momentum 0.9 --decay-lr 30 --pretrained


# lr = 0.001

# Resnet152 -------------------------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder ../datasets/KMNIST --model-name resnet152 --epochs 100 --experiment-name classification \
    --output-folder ../output --ignoregit --lr 0.001 --momentum 0.9 --decay-lr 30

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder ../datasets/KMNIST --model-name resnet152 --epochs 100 --experiment-name classification \
    --output-folder ../output --ignoregit --lr 0.001 --momentum 0.9 --decay-lr 30 --pretrained

# VGG19 -------------------------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder ../datasets/KMNIST --model-name vgg19_bn --epochs 100 --experiment-name classification \
    --output-folder ../output --ignoregit --lr 0.001 --momentum 0.9 --decay-lr 30

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder ../datasets/KMNIST --model-name vgg19_bn --epochs 100 --experiment-name classification \
    --output-folder ../output --ignoregit --lr 0.001 --momentum 0.9 --decay-lr 30 --pretrained

# densenet121 -------------------------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder ../datasets/KMNIST --model-name densenet121 --epochs 100 --experiment-name classification \
    --output-folder ../output --ignoregit --lr 0.001 --momentum 0.9 --decay-lr 30

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder ../datasets/KMNIST --model-name densenet121 --epochs 100 --experiment-name classification \
    --output-folder ../output --ignoregit --lr 0.001 --momentum 0.9 --decay-lr 30 --pretrained

# inception -------------------------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder ../datasets/KMNIST --model-name inception_v3 --epochs 100 --experiment-name classification \
    --output-folder ../output --ignoregit --lr 0.001 --momentum 0.9 --decay-lr 30

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder ../datasets/KMNIST --model-name inception_v3 --epochs 100 --experiment-name classification \
    --output-folder ../output --ignoregit --lr 0.001 --momentum 0.9 --decay-lr 30 --pretrained
