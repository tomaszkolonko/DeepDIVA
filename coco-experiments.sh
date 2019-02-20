#!/bin/bash

python template/RunMe.py --runner-class semantic_segmentation_coco --output-folder /home/linda/output --dataset-folder /data/coco2017 --ignoregit --experiment-name testcoco --epoch 50 --model-name fcdensenet57 --no-val-conf-matrix --batch-size 4 --disable-databalancing --lr 0.01 --momentum 0.9 -j 8 --gpu-id 0