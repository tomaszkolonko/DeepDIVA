#!/bin/bash

#Experiments for HisDB-55

# Unet ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /home/ltp/dataset/HisDB/CB55/ --model-name Unet --epochs 30 --experiment-name ls-sigopt-segm-Unet --output-folder ../output/ --decay-lr 15 --ignoregit --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 10 --sig-opt util/sigopt.json --batch-size 6 --crop-size 256 --pages-in-memory 3 --crops-per-page 200


# fcdensenet67 ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /home/ltp/dataset/HisDB/CB55/ --model-name fcdensenet57 --epochs 30 --experiment-name ls-sigopt-segm-fcdensenet57 --output-folder ../output/ --decay-lr 15 --ignoregit --sig-opt-token LVVXZLFGZSCTCTAXCIXXTWFZUFLYSLRCIUGEZXAMXRXAXYGD --sig-opt-runs 10 --sig-opt util/sigopt.json --batch-size 6 --crop-size 256 --pages-in-memory 3 --crops-per-page 200