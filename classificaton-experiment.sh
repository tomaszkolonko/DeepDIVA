#!/bin/bash

# Resnet18 ---------------------------------------
python ./template/RunMe.py --runner-class image_classification --dataset-folder ../datasets/ICDAR2017-CLAMM/StyleClassification/ 
--model-name resnet18 --epochs 50 --experiment-name classification --output-folder ../output/ --ignoregit --lr 0.01 --momentum 0.9 
--decay-lr 20

python ./template/RunMe.py --runner-class image_classification --dataset-folder ../datasets/ICDAR2017-CLAMM/StyleClassification/ 
--model-name resnet18 --epochs 50 --experiment-name classification --output-folder ../output/ --ignoregit --lr 0.01 --momentum 0.9 
--decay-lr 20 --pretrained

# VGG19 ---------------------------------------
python ./template/RunMe.py --runner-class image_classification --dataset-folder ../datasets/ICDAR2017-CLAMM/StyleClassification/ 
--model-name vgg19 --epochs 50 --experiment-name classification --output-folder ../output/ --ignoregit --lr 0.01 --momentum 0.9 
--decay-lr 20

python ./template/RunMe.py --runner-class image_classification --dataset-folder ../datasets/ICDAR2017-CLAMM/StyleClassification/
--model-name vgg19 --epochs 50 --experiment-name classification --output-folder ../output/ --ignoregit --lr 0.01 --momentum 0.9
--decay-lr 20 --pretrained

# VGG19_bn ---------------------------------------
python ./template/RunMe.py --runner-class image_classification --dataset-folder ../datasets/ICDAR2017-CLAMM/StyleClassification/
--model-name vgg19_bn --epochs 50 --experiment-name classification --output-folder ../output/ --ignoregit --lr 0.01 --momentum 0.9
--decay-lr 20

python ./template/RunMe.py --runner-class image_classification --dataset-folder ../datasets/ICDAR2017-CLAMM/StyleClassification/
--model-name vgg19_bn --epochs 50 --experiment-name classification --output-folder ../output/ --ignoregit --lr 0.01 --momentum 0.9
--decay-lr 20 --pretrained

# Densenet121 ---------------------------------------
python ./template/RunMe.py --runner-class image_classification --dataset-folder ../datasets/ICDAR2017-CLAMM/StyleClassification/
--model-name densenet121 --epochs 50 --experiment-name classification --output-folder ../output/ --ignoregit --lr 0.01 --momentum 
0.9
--decay-lr 20

python ./template/RunMe.py --runner-class image_classification --dataset-folder ../datasets/ICDAR2017-CLAMM/StyleClassification/
--model-name 121 --epochs 50 --experiment-name classification --output-folder ../output/ --ignoregit --lr 0.01 --momentum 0.9
--decay-lr 20 --pretrained


