#!/bin/bash

# Resnet152 ---------------------------------------
python ./template/RunMe.py --runner-class image_classification --dataset-folder ../dataset/ICDAR2017-CLAMM/ManuscrpitDating/ \
 --model-name resnet152 --epochs 50 --experiment-name classification-manu-resnet --output-folder ../classification-manu/ --ignoregit --lr 0.07116 --momentum 0.49010 \
 --multi-crop 10 --decay-lr 20 --weight-decay 0.01000

python ./template/RunMe.py --runner-class image_classification --dataset-folder ../dataset/ICDAR2017-CLAMM/ManuscrpitDating/ \
 --model-name resnet152 --epochs 50 --experiment-name classification-manu-resnet --output-folder ../classification-manu/ --ignoregit --lr 0.01949 --momentum 0.56291 \
 --multi-crop 10 --decay-lr 20 --weight-decay 0.00498 --pretrained

# VGG19_bn ---------------------------------------
#python ./template/RunMe.py --runner-class image_classification --dataset-folder ../dataset/ICDAR2017-CLAMM/ManuscrpitDating/ \
# --model-name vgg19_bn --epochs 50 --experiment-name classification-manu-vgg --output-folder ../classification-manu/ --ignoregit \
# --multi-crop 10 --lr 0.04509 --momentum 0.11104 --weight-decay 0.00680 --decay-lr 20

#python ./template/RunMe.py --runner-class image_classification --dataset-folder ../dataset/ICDAR2017-CLAMM/ManuscrpitDating/ \
# --model-name vgg19_bn --epochs 50 --experiment-name classification-manu-vgg --output-folder ../classification-manu/ --ignoregit \
# --multi-crop 10 --lr 0.09656 --momentum 0.65385 --weight-decay 0.00435 --decay-lr 20 --pretrained

# Densenet121 ---------------------------------------
python ./template/RunMe.py --runner-class image_classification --dataset-folder ../dataset/ICDAR2017-CLAMM/ManuscrpitDating/ \
 --model-name densenet121 --epochs 50 --experiment-name classification-manu-densenet --output-folder ../classification-manu/ --ignoregit \
 --multi-crop 10 --lr 0.04435 --momentum 0.34480 --weight-decay 0.00848 --decay-lr 20

python ./template/RunMe.py --runner-class image_classification --dataset-folder ../dataset/ICDAR2017-CLAMM/ManuscrpitDating/ \
 --model-name densenet121 --epochs 50 --experiment-name classification-manu-densenet --output-folder ../classification-manu/ --ignoregit \
 --multi-crop 10 --lr 0.03187 --momentum 0.20721 --weight-decay 0.00006 --decay-lr 20 --pretrained

#InceptionV3
python ./template/RunMe.py --runner-class image_classification --dataset-folder ../dataset/ICDAR2017-CLAMM/ManuscrpitDating/ \
 --model-name inception_v3 --epochs 50 --experiment-name classification-manu-inception --output-folder ../classification-manu/ --ignoregit \
 --multi-crop 10 --lr 0.01523 --momentum 0.95131 --weight-decay 0.00674 --decay-lr 20

python ./template/RunMe.py --runner-class image_classification --dataset-folder ../dataset/ICDAR2017-CLAMM/ManuscrpitDating/ \
 --model-name inception_v3 --epochs 50 --experiment-name classification-manu-inception --output-folder ../classification-manu/ --ignoregit \
 --multi-crop 10 --lr 0.01788 --momentum 0.81922 --weight-decay 0.00621 --decay-lr 20 --pretrained