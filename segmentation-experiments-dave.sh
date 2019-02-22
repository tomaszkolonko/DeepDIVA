#!/bin/bash

#Experiments for HisDB-55

# SegNet ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets//HisDB-private/CB55/ --model-name SegNet --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix-j 32

python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets//HisDB-private/CB55/ --model-name SegNet --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix-j 32 --pretrained

#Experiments for HisDB-CS863

# SegNet ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets//HisDB-private/CS863/ --model-name SegNet --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix-j 32

python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets//HisDB-private/CS863/ --model-name SegNet --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix-j 32 --pretrained


#Experiments for HisDB-CS18

# SegNet ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets//HisDB-private/CS18/ --model-name SegNet --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix-j 32

python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder ../datasets//HisDB-private/CS18/ --model-name SegNet --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix-j 32 --pretrained
