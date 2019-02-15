#!/bin/bash

#Experiments for HisDB-55

# fcdensenet57 ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets/HisDB-private/CB55/ --model-name fcdensenet57 --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 8 --crop-size 256 --pages-in-memory 3 --crops-per-page 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix


#Experiments for HisDB-CS863

# fcdensenet57 ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets/HisDB-private/CS863/ --model-name fcdensenet57 --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 8 --crop-size 256 --pages-in-memory 3 --crops-per-page 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix


#Experiments for HisDB-CS18
# Unet ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets/HisDB-private/CS18/ --model-name Unet --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 --ignoregit \
 --batch-size 32 -j 8 --crop-size 256 --pages-in-memory 3 --crops-per-page 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix

# fcdensenet57 ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets/HisDB-private/CS18/ --model-name fcdensenet57 --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 8 --crop-size 256 --pages-in-memory 3 --crops-per-page 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix



exit 1

#Experiments for HisDB-CS863
# Unet ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets/HisDB-private/CS863/ --model-name Unet --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 --ignoregit \
 --batch-size 32 -j 8 --crop-size 256 --pages-in-memory 3 --crops-per-page 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix


#Experiments for HisDB-55
# Unet ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets/HisDB-private/CB55/ --model-name Unet --epochs 50 --experiment-name segmentation-normaldist --output-folder ../output/ --decay-lr 24 --ignoregit \
 --batch-size 32 -j 8 --crop-size 256 --pages-in-memory 3 --crops-per-page 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix


#Experiments for HisDB-55

# Unet ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets/HisDB-private/CB55/ --model-name Unet --epochs 60 --experiment-name segmentation --output-folder ../output/ --decay-lr 28 --ignoregit \
 --batch-size 32 -j 8 --crop-size 256 --pages-in-memory 3 --crops-per-page 1000 --momentum 0.11280 --weight-decay 0.00594 --lr 0.16824 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix

# fcdensenet67 ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets/HisDB-private/CB55/ --model-name fcdensenet57 --epochs 60 --experiment-name segmentation --output-folder ../output/ --decay-lr 28 \
 --batch-size 32 -j 8 --crop-size 256 --pages-in-memory 3 --crops-per-page 1000 --momentum 0.32149 --weight-decay 0.00549 --lr 0.09272 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix