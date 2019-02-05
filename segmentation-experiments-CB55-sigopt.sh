#!/bin/bash

#Experiments for HisDB-CS18

# Unet ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets/HisDB-private/CS18/ --model-name Unet --epochs 30 --experiment-name ls-sigopt-segm-Unet-CS18 --output-folder ../output/ --decay-lr 15 --ignoregit --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 10 --sig-opt util/sigopt.json --batch-size 32 --crop-size 256 --pages-in-memory 3 --crops-per-page 200 -j 8

# fcdensenet67 ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets/HisDB-private/CS18/ --model-name fcdensenet57 --epochs 30 --experiment-name ls-sigopt-segm-fcdensenet57-CS18 --output-folder ../output/ --decay-lr 15 --ignoregit --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 10 --sig-opt util/sigopt.json --batch-size 32 --crop-size 256 --pages-in-memory 3 --crops-per-page 200 -j 8


#Experiments for HisDB-CS863

# Unet ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets/HisDB-private/CS863/ --model-name Unet --epochs 30 --experiment-name ls-sigopt-segm-Unet-CS863 --output-folder ../output/ --decay-lr 15 --ignoregit --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 10 --sig-opt util/sigopt.json --batch-size 32 --crop-size 256 --pages-in-memory 3 --crops-per-page 200 -j 8

# fcdensenet67 ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets/HisDB-private/CS863/ --model-name fcdensenet57 --epochs 30 --experiment-name ls-sigopt-segm-fcdensenet57-CS863 --output-folder ../output/ --decay-lr 15 --ignoregit --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 10 --sig-opt util/sigopt.json --batch-size 32 --crop-size 256 --pages-in-memory 3 --crops-per-page 200 -j 8


#Experiments for HisDB-55

# Unet ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets/HisDB/CB55/ --model-name Unet --epochs 30 --experiment-name ls-sigopt-segm-Unet --output-folder ../output/ --decay-lr 15 --ignoregit --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 10 --sig-opt util/sigopt.json --batch-size 32 --crop-size 256 --pages-in-memory 3 --crops-per-page 200 -j 8

# fcdensenet67 ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets/HisDB/CB55/ --model-name fcdensenet57 --epochs 30 --experiment-name ls-sigopt-segm-fcdensenet57 --output-folder ../output/ --decay-lr 15 --ignoregit --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 10 --sig-opt util/sigopt.json --batch-size 32 --crop-size 256 --pages-in-memory 3 --crops-per-page 200 -j 8
