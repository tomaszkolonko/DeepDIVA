#!/bin/bash

# MANUSCRIPT DATING
# ==================

# Resnet152 -------------------------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/ltp/dataset/ICDAR2017-CLAMM/ManuscrpitDating_nv \
    --model-name resnet152 --epochs 200 --experiment-name tz_long_manuscript_nv_resnet152 \
    --output-folder ../output_tz_classification_manuscript_long/ --ignoregit --lr 0.07116 \
    --momentum 0.49010 --decay-lr 60 --weight-decay 0.01000

# Resnet152 with pre-training ---------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/ltp/dataset/ICDAR2017-CLAMM/ManuscrpitDating_nv \
    --model-name resnet152 --epochs 200 --experiment-name tz_long_manuscript_nv_resnet152_pre \
    --output-folder ../output_tz_classification_manuscript_long/ --ignoregit --lr 0.01949 --pretrained \
    --momentum 0.56291 --decay-lr 60 --weight-decay 0.00498

# STYLE CLASSIFICATION
# =====================

# VGG19_bn -------------------------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/ltp/dataset/ICDAR2017-CLAMM/StyleClassification_nv \
    --model-name vgg19_bn --epochs 200 --experiment-name tz_long_style_nv_vgg19 \
    --output-folder ../output_tz_classification_style_long/ --ignoregit --lr 0.04509 \
    --momentum 0.11104 --decay-lr 60 --weight-decay 0.00680

# VGG19_bn with pre-training ---------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/ltp/dataset/ICDAR2017-CLAMM/StyleClassification_nv \
    --model-name vgg19_bn --epochs 200 --experiment-name tz_long_style_nv_vgg19_pre \
    --output-folder ../output_tz_classification_style_long/ --ignoregit --lr 0.09656 --pretrained \
    --momentum 0.65385 --decay-lr 60 --weight-decay 0.00435