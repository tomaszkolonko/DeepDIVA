#!/bin/bash


# CB55
python template/RunMe.py --runner-class semantic_segmentation_hisdb --output-folder /home/linda/output/ \
 --dataset-folder /home/linda/datasets/HisDB/CB55 --ignoregit --experiment-name segmentation --crop-size 256 --pages-in-memory 3 \
 --crops-per-page 50 --batch-size 8 --epoch 50 --model-name Unet

python template/RunMe.py --runner-class semantic_segmentation_hisdb --output-folder /home/linda/output/ \
 --dataset-folder /home/linda/datasets/HisDB/CB55 --ignoregit --experiment-name segmentation --crop-size 256 --pages-in-memory 3 \
 --crops-per-page 50 --batch-size 8 --epoch 50 --model-name fcdensenet57

# CS18
python template/RunMe.py --runner-class semantic_segmentation_hisdb --output-folder /home/linda/output/ \ 
 --dataset-folder /home/linda/datasets/HisDB/CS18 --ignoregit --experiment-name segmentation --crop-size 256 --pages-in-memory 3 \
 --crops-per-page 50 --batch-size 8 --epoch 50 --model-name Unet

python template/RunMe.py --runner-class semantic_segmentation_hisdb --output-folder /home/linda/output/ \ 
 --dataset-folder /home/linda/datasets/HisDB/CS18 --ignoregit --experiment-name segmentation --crop-size 256 --pages-in-memory 3 \
 --crops-per-page 50 --batch-size 8 --epoch 50 --model-name fcdensenet57

python template/RunMe.py --runner-class semantic_segmentation_hisdb --output-folder /home/linda/output/ \ 
 --dataset-folder /home/linda/datasets/HisDB/CS18 --ignoregit --experiment-name segmentation --crop-size 256 --pages-in-memory 3 \
 --crops-per-page 50 --batch-size 8 --epoch 50 --model-name fcdensenet67

 # CS18
python template/RunMe.py --runner-class semantic_segmentation_hisdb --output-folder /home/linda/output/ \
 --dataset-folder /home/linda/datasets/HisDB/CS18 --ignoregit --experiment-name segmentation --crop-size 256 --pages-in-memory 3 \
 --crops-per-page 50 --batch-size 8 --epoch 50 --model-name Unet

python template/RunMe.py --runner-class semantic_segmentation_hisdb --output-folder /home/linda/output/ \
 --dataset-folder /home/linda/datasets/HisDB/CS18 --ignoregit --experiment-name segmentation --crop-size 256 --pages-in-memory 3 \
 --crops-per-page 50 --batch-size 8 --epoch 50 --model-name fcdensenet57

python template/RunMe.py --runner-class semantic_segmentation_hisdb --output-folder /home/linda/output/ \
 --dataset-folder /home/linda/datasets/HisDB/CS18 --ignoregit --experiment-name segmentation --crop-size 256 --pages-in-memory 3 \
 --crops-per-page 50 --batch-size 8 --epoch 50 --model-name fcdensenet67

 # CS863
python template/RunMe.py --runner-class semantic_segmentation_hisdb --output-folder /home/linda/output/ \
 --dataset-folder /home/linda/datasets/HisDB/CS863 --ignoregit --experiment-name segmentation --crop-size 256 --pages-in-memory 3 \
 --crops-per-page 50 --batch-size 8 --epoch 50 --model-name Unet

python template/RunMe.py --runner-class semantic_segmentation_hisdb --output-folder /home/linda/output/ \
 --dataset-folder /home/linda/datasets/HisDB/CS863 --ignoregit --experiment-name segmentation --crop-size 256 --pages-in-memory 3 \
 --crops-per-page 50 --batch-size 8 --epoch 50 --model-name fcdensenet57

python template/RunMe.py --runner-class semantic_segmentation_hisdb --output-folder /home/linda/output/ \
 --dataset-folder /home/linda/datasets/HisDB/CS863 --ignoregit --experiment-name segmentation --crop-size 256 --pages-in-memory 3 \
 --crops-per-page 50 --batch-size 8 --epoch 50 --model-name fcdensenet67


