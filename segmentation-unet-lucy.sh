#!/bin/bash

#Experiments for HisDB-55

# unet pretrained ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CB55/ --model-name unet --epochs 50 --experiment-name test --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 2 -j 32 --pretrained \
 --path-pretrained-model /home/linda/output/coco/coco2017/model_name\=Unet/batch_size\=16/epochs\=50/lr\=0.02/momentum\=0.9/gpu_id\=0/workers\=16/no_val_conf_matrix\=True/resize_coco\=320/20-02-19-13h-29m-32s/model_best.pth.tar


#Experiments for HisDB-CS863

# unet pretrained ---------------------------------------

python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CS863/ --model-name unet --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 4 -j 32 --pretrained \
 --path-pretrained-model /home/linda/output/coco/coco2017/model_name\=Unet/batch_size\=16/epochs\=50/lr\=0.02/momentum\=0.9/gpu_id\=0/workers\=16/no_val_conf_matrix\=True/resize_coco\=320/20-02-19-13h-29m-32s/model_best.pth.tar


#Experiments for HisDB-CS18

# unet pretrained ---------------------------------------

python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CS18/ --model-name unet --epochs 50 --experiment-name test --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 5 -j 32 --pretrained \
 --path-pretrained-model /home/linda/output/coco/coco2017/model_name\=Unet/batch_size\=16/epochs\=50/lr\=0.02/momentum\=0.9/gpu_id\=0/workers\=16/no_val_conf_matrix\=True/resize_coco\=320/20-02-19-13h-29m-32s/model_best.pth.tar
