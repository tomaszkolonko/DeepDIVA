#!/bin/bash

#Experiments for HisDB-55

# fcdensenet57 pretrained ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CB55/ --model-name fcdensenet57 --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 8 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 0 -j 32 --pretrained \
 --path-pretrained-model /home/linda/output/coco/coco2017/model_name=fcdensenet57/batch_size=16/epochs=50/lr=0.01/momentum=0.9/gpu_id=0/workers=16/no_val_conf_matrix=True/resize_coco=320/22-02-19-14h-49m-36s/model_best.pth.tar


#Experiments for HisDB-CS863

# fcdensenet57 pretrained ---------------------------------------

python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CS863/ --model-name fcdensenet57 --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 8 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 0 -j 32 --pretrained \
 --path-pretrained-model /home/linda/output/coco/coco2017/model_name=fcdensenet57/batch_size=16/epochs=50/lr=0.01/momentum=0.9/gpu_id=0/workers=16/no_val_conf_matrix=True/resize_coco=320/22-02-19-14h-49m-36s/model_best.pth.tar


#Experiments for HisDB-CS18

# fcdensenet57 pretrained ---------------------------------------

python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CS18/ --model-name fcdensenet57 --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 8 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 0 -j 32 --pretrained \
 --path-pretrained-model /home/linda/output/coco/coco2017/model_name=fcdensenet57/batch_size=16/epochs=50/lr=0.01/momentum=0.9/gpu_id=0/workers=16/no_val_conf_matrix=True/resize_coco=320/22-02-19-14h-49m-36s/model_best.pth.tar
