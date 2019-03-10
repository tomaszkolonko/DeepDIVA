#!/bin/bash

# CNN_basic
python ./template/RunMe.py --runner-class image_classification --dataset-folder ../datasets/ICDAR2017-CLAMM/StyleClassification/ \
 --model-name cnn_basic --epochs 50 --experiment-name classification-style-resnet --output-folder ../classification-style/ --ignoregit --lr 0.06221 --momentum 0.54338 \
 --multi-crop 10 --decay-lr 20 --weight-decay 0.00771

# Resnet152 ---------------------------------------
#python ./template/RunMe.py --runner-class image_classification --dataset-folder ../datasets/ICDAR2017-CLAMM/StyleClassification/ \
# --model-name resnet152 --epochs 50 --experiment-name classification-style-resnet --output-folder ../classification-style/ --ignoregit --lr 0.06221 --momentum 0.54338 \
# --multi-crop 10 --decay-lr 20 --weight-decay 0.00771

#python ./template/RunMe.py --runner-class image_classification --dataset-folder ../datasets/ICDAR2017-CLAMM/StyleClassification/ \
# --model-name resnet152 --epochs 50 --experiment-name classification-style-resnet --output-folder ../classification-style/ --ignoregit --lr 0.02609 --momentum 0.74250 \
# --multi-crop 10 --decay-lr 20 --weight-decay 0.00895 --pretrained

# VGG19_bn ---------------------------------------
#python ./template/RunMe.py --runner-class image_classification --dataset-folder ../datasets/ICDAR2017-CLAMM/StyleClassification/ \
# --model-name vgg19_bn --epochs 50 --experiment-name classification-style-vgg --output-folder ../classification-style/ --ignoregit \
# --multi-crop 10 --lr 0.04509 --momentum 0.11104 --weight-decay 0.00680 --decay-lr 20

#python ./template/RunMe.py --runner-class image_classification --dataset-folder ../datasets/ICDAR2017-CLAMM/StyleClassification/ \
# --model-name vgg19_bn --epochs 50 --experiment-name classification-style-vgg --output-folder ../classification-style/ --ignoregit \
# --multi-crop 10 --lr 0.09656 --momentum 0.65385 --weight-decay 0.00435 --decay-lr 20 --pretrained

# Densenet121 ---------------------------------------
#python ./template/RunMe.py --runner-class image_classification --dataset-folder ../datasets/ICDAR2017-CLAMM/StyleClassification/ \
# --model-name densenet121 --epochs 50 --experiment-name classification-style-densenet --output-folder ../classification-style/ --ignoregit \
# --multi-crop 10 --lr 0.08750 --momentum 0.32541 --weight-decay 0.00389 --decay-lr 20

#python ./template/RunMe.py --runner-class image_classification --dataset-folder ../datasets/ICDAR2017-CLAMM/StyleClassification/ \
# --model-name densenet121 --epochs 50 --experiment-name classification-style-densenet --output-folder ../classification-style/ --ignoregit \
# --multi-crop 10 --lr 0.05002 --momentum 0.42603 --weight-decay 0.00575 --decay-lr 20 --pretrained

#InceptionV3
#python ./template/RunMe.py --runner-class image_classification --dataset-folder ../datasets/ICDAR2017-CLAMM/StyleClassification/ \
# --model-name inception_v3 --epochs 50 --experiment-name classification-style-inception --output-folder ../classification-style/ --ignoregit \
# --multi-crop 10 --lr 0.08231 --momentum 0.67468 --weight-decay 0.00689 --decay-lr 20

#python ./template/RunMe.py --runner-class image_classification --dataset-folder ../datasets/ICDAR2017-CLAMM/StyleClassification/ \
# --model-name inception_v3 --epochs 50 --experiment-name classification-style-inception --output-folder ../classification-style/ --ignoregit \
## --multi-crop 10 --lr 0.08782 --momentum 0.25135 --weight-decay 0.00956 --decay-lr 20 --pretrained