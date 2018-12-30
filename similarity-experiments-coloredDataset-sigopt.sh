#!/bin/bash

#Experiments for Colored Dataset NOT PRETRAINED
# Densenet121 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name densenet121 --epochs 5 --experiment-name ls-sigopt-similarity-randomcrop --output-folder ../output/ --decay-lr 25 --n-triplets 50000 --output-channels 128 --map auto --ignoregit --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 5 --sig-opt util/sigopt.json --batch-size 32 --regenerate-every 2

#InceptionV3
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name inception_v3 --epochs 5 --experiment-name ls-sigopt-similarity-randomcrop --output-folder ../output/ --decay-lr 25 --n-triplets 50000 --output-channels 128 --map auto --ignoregit --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 5 --sig-opt util/sigopt.json --batch-size 32 --regenerate-every 2

# VGG19_bn ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name vgg19_bn --epochs 5 --experiment-name ls-sigopt-similarity-randomcrop --output-folder ../output/ --decay-lr 25 --n-triplets 50000 --output-channels 128 --map auto --ignoregit --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 5 --sig-opt util/sigopt.json --batch-size 32 --regenerate-every 2

# Resnet152 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name resnet152 --epochs 5 --experiment-name ls-sigopt-similarity-randomcrop --output-folder ../output/ --decay-lr 25 --n-triplets 50000 --output-channels 128 --map auto --ignoregit --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 5 --sig-opt util/sigopt.json --batch-size 32 --regenerate-every 2


exit 1
#*********************************************************************************************
#Experiments for Colored Dataset PRETRAINED

# Densenet121 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name densenet121 --epochs 5 --experiment-name ls-sigopt-similarity-randomcrop --output-folder ../output/ --decay-lr 20 --n-triplets 50000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 2 --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 5 --sig-opt util/sigopt.json --batch-size 32

#InceptionV3
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name inception_v3 --epochs 5 --experiment-name ls-sigopt-similarity-randomcrop --output-folder ../output/ --decay-lr 20 --n-triplets 50000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 2 --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 5 --sig-opt util/sigopt.json --batch-size 32

# VGG19_bn ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name vgg19_bn --epochs 5 --experiment-name ls-sigopt-similarity-randomcrop --output-folder ../output/ --decay-lr 20 --n-triplets 50000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 2 --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 5 --sig-opt util/sigopt.json --batch-size 32

# Resnet152 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name resnet152 --epochs 5 --experiment-name ls-sigopt-similarity-randomcrop --output-folder ../output/ --decay-lr 20 --n-triplets 50000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 2 --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 5 --sig-opt util/sigopt.json --batch-size 32