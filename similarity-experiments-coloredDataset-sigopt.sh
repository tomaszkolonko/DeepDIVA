#!/bin/bash

#Experiments for Colored Dataset NOT PRETRAINED

# Densenet121 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name densenet121 --epochs 30 --experiment-name ls-sigopt-similarity-randomcrop --output-folder ../output/ --decay-lr 25 --n-triplets 100000 --output-channels 128 --map auto --ignoregit --sig-opt-token KGCBTCPDIXWSRHAGJVIHPWGGJTPEBXCWFETOQIYJHEDEOYUF --sig-opt-runs 30 --sig-opt util/sigopt.json --batch-size 32

#InceptionV3
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name inception_v3 --epochs 30 --experiment-name ls-sigopt-similarity-randomcrop --output-folder ../output/ --decay-lr 25 --n-triplets 100000 --output-channels 128 --map auto --ignoregit --sig-opt-token KGCBTCPDIXWSRHAGJVIHPWGGJTPEBXCWFETOQIYJHEDEOYUF --sig-opt-runs 30 --sig-opt util/sigopt.json --batch-size 32

# VGG19_bn ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name vgg19_bn --epochs 30 --experiment-name ls-sigopt-similarity-randomcrop --output-folder ../output/ --decay-lr 25 --n-triplets 100000 --output-channels 128 --map auto --ignoregit --sig-opt-token KGCBTCPDIXWSRHAGJVIHPWGGJTPEBXCWFETOQIYJHEDEOYUF --sig-opt-runs 30 --sig-opt util/sigopt.json --batch-size 32

# Resnet152 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name resnet152 --epochs 30 --experiment-name ls-sigopt-similarity-randomcrop --output-folder ../output/ --decay-lr 25 --n-triplets 100000 --output-channels 128 --map auto --ignoregit --sig-opt-token KGCBTCPDIXWSRHAGJVIHPWGGJTPEBXCWFETOQIYJHEDEOYUF --sig-opt-runs 30 --sig-opt util/sigopt.json --batch-size 32


exit 1
#*********************************************************************************************
#Experiments for Colored Dataset PRETRAINED

# Densenet121 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name densenet121 --epochs 10 --experiment-name ls-sigopt-similarity-randomcrop --output-folder ../output/ --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 2 --sig-opt-token KGCBTCPDIXWSRHAGJVIHPWGGJTPEBXCWFETOQIYJHEDEOYUF --sig-opt-runs 30 --sig-opt util/sigopt.json --batch-size 32

#InceptionV3
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name inception_v3 --epochs 10 --experiment-name ls-sigopt-similarity-randomcrop --output-folder ../output/ --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 2 --sig-opt-token KGCBTCPDIXWSRHAGJVIHPWGGJTPEBXCWFETOQIYJHEDEOYUF --sig-opt-runs 30 --sig-opt util/sigopt.json --batch-size 32

# VGG19_bn ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name vgg19_bn --epochs 10 --experiment-name ls-sigopt-similarity-randomcrop --output-folder ../output/ --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 2 --sig-opt-token KGCBTCPDIXWSRHAGJVIHPWGGJTPEBXCWFETOQIYJHEDEOYUF --sig-opt-runs 30 --sig-opt util/sigopt.json --batch-size 32

# Resnet152 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name resnet152 --epochs 10 --experiment-name ls-sigopt-similarity-randomcrop --output-folder ../output/ --decay-lr 20 --n-triplets 100000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 2 --sig-opt-token KGCBTCPDIXWSRHAGJVIHPWGGJTPEBXCWFETOQIYJHEDEOYUF --sig-opt-runs 30 --sig-opt util/sigopt.json --batch-size 32