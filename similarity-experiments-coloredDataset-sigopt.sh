#!/bin/bash

#Experiments for Colored Dataset NOT PRETRAINED


#*********************************************************************************************
#Experiments for Colored Dataset PRETRAINED

# VGG19_bn ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name vgg19_bn --epochs 5 --experiment-name ls-sigopt-similarity-randomcrop-pretrained-vgg19_bn --output-folder ../output/ --decay-lr 20 --n-triplets 50000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 1 --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 5 --sig-opt util/sigopt-simpretrained.json --batch-size 32

# Resnet152 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name resnet152 --epochs 5 --experiment-name ls-sigopt-similarity-randomcrop-pretrained-resnet152 --output-folder ../output/ --decay-lr 20 --n-triplets 50000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 1 --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 5 --sig-opt util/sigopt-simpretrained.json --batch-size 32

#InceptionV3
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name inception_v3 --epochs 5 --experiment-name ls-sigopt-similarity-randomcrop-pretrained-inception_v3 --output-folder ../output/ --decay-lr 20 --n-triplets 50000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 1 --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 10 --sig-opt util/sigopt-simpretrained.json --batch-size 32

# Densenet121 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name densenet121 --epochs 5 --experiment-name ls-sigopt-similarity-randomcrop-pretrained-densenet121 --output-folder ../output/ --decay-lr 20 --n-triplets 50000 --output-channels 128 --map auto --pretrained --ignoregit --regenerate-every 1 --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 10 --sig-opt util/sigopt-simpretrained.json --batch-size 32

exit 1

# VGG19_bn ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name vgg19_bn --epochs 5 --experiment-name ls-sigopt-similarity-randomcrop-vgg19_bn --output-folder ../output/ --decay-lr 25 --n-triplets 50000 --output-channels 128 --map auto --ignoregit --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 5 --sig-opt util/sigopt.json --batch-size 32--regenerate-every 1

#InceptionV3
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name inception_v3 --epochs 5 --experiment-name ls-sigopt-similarity-randomcrop-inception_v3 --output-folder ../output/ --decay-lr 25 --n-triplets 50000 --output-channels 128 --map auto --ignoregit --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 5 --sig-opt util/sigopt.json --batch-size 32--regenerate-every 1

# Resnet152 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name resnet152 --epochs 5 --experiment-name ls-sigopt-similarity-randomcrop-resnet152 --output-folder ../output/ --decay-lr 25 --n-triplets 50000 --output-channels 128 --map auto --ignoregit --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 5 --sig-opt util/sigopt.json --batch-size 32--regenerate-every 1

# Densenet121 ---------------------------------------
python ./template/RunMe.py --runner-class triplet --dataset-folder ../datasets/historical_wi_col_small/ --model-name densenet121 --epochs 5 --experiment-name ls-sigopt-similarity-randomcrop-densenet121 --output-folder ../output/ --decay-lr 25 --n-triplets 50000 --output-channels 128 --map auto --ignoregit --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 5 --sig-opt util/sigopt.json --batch-size 32--regenerate-every 1
