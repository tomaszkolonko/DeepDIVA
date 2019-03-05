#!/bin/bash


#----------------1/10 HISDB MULTIRUN--------------------------------------------------------------------------------------------------------------------------
#Experiments for HisDB-55

# SegNet ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/HisDB_tenth_resolution/CB55/ --model-name SegNet --epochs 50 --experiment-name segmentation-multi --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 50 --momentum 0.9 --lr 0.001 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 2 -j 32 --multi-run 5 &

python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/HisDB_tenth_resolution/CB55/ --model-name SegNet --epochs 50 --experiment-name segmentation-multi --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 50 --momentum 0.9 --lr 0.001 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 3 -j 32 --pretrained --multi-run 5


#Experiments for HisDB-CS863

# SegNet ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/HisDB_tenth_resolution/CSG863/ --model-name SegNet --epochs 50 --experiment-name segmentation-multi --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 50 --momentum 0.9 --lr 0.001 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 2 -j 32 --pretrained &

python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/HisDB_tenth_resolution/CSG863/ --model-name SegNet --epochs 50 --experiment-name segmentation-multi --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 50 --momentum 0.9 --lr 0.001 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 3 -j 32 --multi-run 5


#Experiments for HisDB-CS18

# SegNet ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/HisDB_tenth_resolution/CSG18/ --model-name SegNet --epochs 50 --experiment-name segmentation-multi --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 50 --momentum 0.9 --lr 0.001 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 2 -j 32 &

python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/HisDB_tenth_resolution/CSG18/ --model-name SegNet --epochs 50 --experiment-name segmentation-multi --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 50 --momentum 0.9 --lr 0.001 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 3 -j 32 --pretrained


exit 1
#------------------------------------------------------------------------------------------------------------------------------------------


#Experiments for HisDB-55

# SegNet ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CB55/ --model-name SegNet --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.0005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 7 -j 32

python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CB55/ --model-name SegNet --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.0005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 7 -j 32 --pretrained


#Experiments for HisDB-CS863

# SegNet ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CS863/ --model-name SegNet --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.0005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 7 -j 32 --pretrained

python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CS863/ --model-name SegNet --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.0005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 7 -j 32


#Experiments for HisDB-CS18

# SegNet ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CS18/ --model-name SegNet --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.0005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 7 -j 32

python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CS18/ --model-name SegNet --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.0005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 7 -j 32 --pretrained

exit 1

#******************************************************************************************************************
#******************************************************************************************************************

#Experiments for HisDB-55

# deeplabv3 ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CB55/ --model-name deeplabv3 --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 3 -j 32 --pretrained

#Experiments for HisDB-CS863

# deeplabv3 ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CS863/ --model-name deeplabv3 --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 3 -j 32 --pretrained


#Experiments for HisDB-CS18

# deeplabv3 ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CS18/ --model-name deeplabv3 --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 32 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 3 -j 32 --pretrained


#Experiments for HisDB-55

# fcdensenet67 ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CB55/ --model-name fcdensenet57 --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 8 -j 16 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 0


#Experiments for HisDB-CS863

# fcdensenet67 ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CS863/ --model-name fcdensenet57 --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 8 -j 16 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 0


#Experiments for HisDB-CS18

# fcdensenet67 ---------------------------------------
python ./template/RunMe.py --runner-class semantic_segmentation_hisdb --dataset-folder /data/ltp/HisDB-private/CS18/ --model-name fcdensenet57 --epochs 50 --experiment-name segmentation --output-folder ../output/ --decay-lr 24 \
 --ignoregit --batch-size 8 -j 16 --crop-size 256 --imgs-in-memory 3 --crops-per-image 1000 --momentum 0.9 --lr 0.005 \
 --disable-databalancing --use-boundary-pixel --no-val-conf-matrix --gpu-id 0
