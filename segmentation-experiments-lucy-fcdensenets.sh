#!/bin/bash

#Experiments for HisDB-55

# fcdensenet67 ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CB55/ --model-name fcdensenet57 --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 -j 16 --crop-size 256 --pages-in-memory 3 --crops-per-page 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 0


#Experiments for HisDB-CS863

# fcdensenet67 ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CS863/ --model-name fcdensenet57 --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 -j 16 --crop-size 256 --pages-in-memory 3 --crops-per-page 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 0


#Experiments for HisDB-CS18

# fcdensenet67 ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CS18/ --model-name fcdensenet57 --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 -j 16 --crop-size 256 --pages-in-memory 3 --crops-per-page 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 0
