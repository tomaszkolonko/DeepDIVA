#Vgg11_bn pretrain=false
python template/RunMe.py --runner-class apply_model --dataset-folder ../../dataset/ICDAR2017-CLAMM/ManuscriptDating/ --load-model ../../results-classification-pc30/classification/ManuscriptDating/model_name\=vgg11_bn/epochs\=50/lr\=0.01/decay_lr\=20/momentum\=0.9/20-11-18-19h-12m-10s/checkpoint.pth.tar --output-channels 15 --ignoregit

python util/visualization/embedding.py --results-file ./output/classify/Manuscriptdating/classify\=True/EXCUTION-DATE/results.pkl --output-file ../../results-classification-pc30/classification/ManuscriptDating/vgg11_bn_md.png --tensorboard

#Vgg11_bn pretrain=true
python template/RunMe.py --runner-class apply_model --dataset-folder ../../dataset/ICDAR2017-CLAMM/ManuscriptDating/ --load-model ../../results-classification-pc30/classification/ManuscriptDating/model_name\=vgg11_bn/epochs\=50/pretrained\=True/lr\=0.01/decay_lr\=20/momentum\=0.9/20-11-18-19h-51m-38s/checkpoint.pth.tar --output-channels 15 --ignoregit

python util/visualization/embedding.py --results-file ./output/classify/Manuscriptdating/classify\=True/EXCUTION-DATE/results.pkl --output-file ../../results-classification-pc30/classification/ManuscriptDating/vgg11_bn_pretrained_md.png --tensorboard

#Vgg19_bn pretrain=false
python template/RunMe.py --runner-class apply_model --dataset-folder ../../dataset/ICDAR2017-CLAMM/ManuscriptDating/ --load-model ../../results-classification-pc30/classification/ManuscriptDating/model_name\=vgg19_bn/epochs\=50/lr\=0.01/decay_lr\=20/momentum\=0.9/19-11-18-22h-15m-13s/checkpoint.pth.tar --output-channels 15 --ignoregit

python util/visualization/embedding.py --results-file ./output/classify/Manuscriptdating/classify\=True/EXCUTION-DATE/results.pkl --output-file ../../results-classification-pc30/classification/ManuscriptDating/vgg19_bn_md.png --tensorboard

#Vgg19_bn pretrain=false
python template/RunMe.py --runner-class apply_model --dataset-folder ../../dataset/ICDAR2017-CLAMM/ManuscriptDating/ --load-model ../../results-classification-pc30/classification/ManuscriptDating/model_name\=vgg19_bn/epochs\=50/pretrained\=True/lr\=0.01/decay_lr\=20/momentum\=0.9/19-11-18-22h-54m-57s/checkpoint.pth.tar --output-channels 15 --ignoregit

python util/visualization/embedding.py --results-file ./output/classify/Manuscriptdating/classify\=True/EXCUTION-DATE/results.pkl --output-file ../../results-classification-pc30/classification/ManuscriptDating/vgg19_bn_pretrained_md.png --tensorboard

#DenseNet201 pretrain=false
python template/RunMe.py --runner-class apply_model --dataset-folder ../../dataset/ICDAR2017-CLAMM/ManuscriptDating/ --load-model ../../results-classification-pc30/classification/ManuscriptDating/model_name\=densenet201/epochs\=50/lr\=0.01/decay_lr\=20/momentum\=0.9/EXCUTION-DATE/checkpoint.pth.tar --output-channels 15 --ignoregit

python util/visualization/embedding.py --results-file ./output/classify/Manuscriptdating/classify\=True/EXCUTION-DATE/results.pkl --output-file ../../results-classification-pc30/classification/ManuscriptDating/vgg11_bn_md.png --tensorboard

#DenseNet201 pretrain=true
python template/RunMe.py --runner-class apply_model --dataset-folder ../../dataset/ICDAR2017-CLAMM/ManuscriptDating/ --load-model ../../results-classification-pc30/classification/ManuscriptDating/model_name\=densenet201/epochs\=50/pretrained\=True/lr\=0.01/decay_lr\=20/momentum\=0.9/EXCUTION-DATE/checkpoint.pth.tar --output-channels 15 --ignoregit

python util/visualization/embedding.py --results-file ./output/classify/Manuscriptdating/classify\=True/EXCUTION-DATE/results.pkl --output-file ../../results-classification-pc30/classification/ManuscriptDating/vgg11_bn_pretrained_md.png --tensorboard
