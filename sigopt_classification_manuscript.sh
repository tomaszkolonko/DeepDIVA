#!/bin/bash

# VGG19_bn ----------------------------------------
python ./template/RunMe.py --runner-class image_classification --dataset-folder /home/thomas.kolonko/datasets/ICDAR2017-CLAMM/ManuscriptDating --model-name vgg19_bn --epochs 50 --experiment-name tz_class_manuscript_vgg --output-folder ../output_tz_classification/ --ignoregit --lr 0.01 --momentum 0.9 --decay-lr 20 --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 30 --sig-opt util/sigopt.json

# Resnet152 ---------------------------------------
python ./template/RunMe.py --runner-class image_classification --dataset-folder /home/thomas.kolonko/datasets/ICDAR2017-CLAMM/ManuscriptDating --model-name resnet152 --epochs 50 --experiment-name tz_class_manuscript_resnet --output-folder ../output_tz_classification/ --ignoregit --lr 0.01 --momentum 0.9 --decay-lr 20 --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 30 --sig-opt util/sigopt.json

# Incecption_v3 -----------------------------------
python ./template/RunMe.py --runner-class image_classification --dataset-folder /home/thomas.kolonko/datasets/ICDAR2017-CLAMM/ManuscriptDating --model-name inception_v3 --epochs 50 --experiment-name tz_class_manuscript_inception --output-folder ../output_tz_classification/ --ignoregit --lr 0.01 --momentum 0.9 --decay-lr 20 --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 30 --sig-opt util/sigopt.json

# Densenet121 -------------------------------------
python ./template/RunMe.py --runner-class image_classification --dataset-folder /home/thomas.kolonko/datasets/ICDAR2017-CLAMM/ManuscriptDating --model-name densenet121 --epochs 50 --experiment-name tz_class_manuscript_densenet --output-folder ../output_tz_classification/ --ignoregit --lr 0.01 --momentum 0.9 --decay-lr 20 --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 30 --sig-opt util/sigopt.json