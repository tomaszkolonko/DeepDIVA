#!/bin/bash

# VGG19_bn ----------------------------------------
python ./template/RunMe.py --runner-class image_classification --dataset-folder /home/.thomas.kolonko/datasets/icdar --model-name vgg19_bn --epochs 50 --experiment-name classification --output-folder ../output_classification/ --ignoregit --lr 0.01 --momentum 0.9 --decay-lr 20 --sig-opt-token KGCBTCPDIXWSRHAGJVIHPWGGJTPEBXCWFETOQIYJHEDEOYUF --sig-opt-runs 30 --sig-opt util/sigopt.json

# Resnet152 ---------------------------------------
python ./template/RunMe.py --runner-class image_classification --dataset-folder /home/.thomas.kolonko/datasets/icdar --model-name resnet152 --epochs 50 --experiment-name classification --output-folder ../output_classification/ --ignoregit --lr 0.01 --momentum 0.9 --decay-lr 20 --sig-opt-token KGCBTCPDIXWSRHAGJVIHPWGGJTPEBXCWFETOQIYJHEDEOYUF --sig-opt-runs 30 --sig-opt util/sigopt.json

# Incecption_v3 -----------------------------------
python ./template/RunMe.py --runner-class image_classification --dataset-folder /home/.thomas.kolonko/datasets/icdar --model-name inception_v3 --epochs 50 --experiment-name classification --output-folder ../output_classification/ --ignoregit --lr 0.01 --momentum 0.9 --decay-lr 20 --sig-opt-token KGCBTCPDIXWSRHAGJVIHPWGGJTPEBXCWFETOQIYJHEDEOYUF --sig-opt-runs 30 --sig-opt util/sigopt.json

# Densenet121 -------------------------------------
python ./template/RunMe.py --runner-class image_classification --dataset-folder /home/.thomas.kolonko/datasets/icdar --model-name densenet121 --epochs 50 --experiment-name classification --output-folder ../output_classification/ --ignoregit --lr 0.01 --momentum 0.9 --decay-lr 20 --sig-opt-token KGCBTCPDIXWSRHAGJVIHPWGGJTPEBXCWFETOQIYJHEDEOYUF --sig-opt-runs 30 --sig-opt util/sigopt.json