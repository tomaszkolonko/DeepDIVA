import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import cv2

import torch

from datasets.transform_library import functional

class CocoDetection(data.Dataset):
    """
    modified version! (original: https://pytorch.org/docs/0.4.0/_modules/torchvision/datasets/coco.html)

    `MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in torch tensor
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, resize_coco=False, name_onehotindex=None, category_id_name=None, **kwargs):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.resize_coco = resize_coco
        self.img_size = max([item for sublist in [[img['height'], img['width']] for img in self.coco.dataset['images']] for item
         in sublist])
        if not name_onehotindex:
            self.name_onehotindex = {d['name']: i + 1 for i, d in enumerate(self.coco.dataset['categories'])}
            self.name_onehotindex['background'] = 0
        else:
            self.name_onehotindex = name_onehotindex
        if not category_id_name:
            self.category_id_name = {d['id']: d['name'] for d in self.coco.dataset['categories']}
        else:
            self.category_id_name = category_id_name

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        # https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros
        resize_coco = self.resize_coco
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        # determine the padding
        top = (self.img_size-img.height)//2
        left = (self.img_size-img.width)//2
        bottom = self.img_size - img.height - top
        right = self.img_size - img.width - left
        padding = (left, top, right, bottom)

        size = (resize_coco, resize_coco)

        # convert the annotations to argmax and pad to make quadratic
        target = functional.annotation_to_argmax((img.height, img.width), target, self.name_onehotindex, self.category_id_name)
        target = np.array(functional.pad(Image.fromarray(np.array(target).astype('uint8')), padding))
        if resize_coco:
            target = torch.LongTensor(cv2.resize(target, dsize=size, interpolation=cv2.INTER_NEAREST))
        else:
            target = torch.LongTensor(target)

        # convert img to torch tensor and pad to make quadratic
        img = functional.pad(img, padding)
        if resize_coco:
            img = cv2.resize(np.asanyarray(img), dsize=size, interpolation=cv2.INTER_NEAREST)
        img = functional.to_tensor(img)

        return img, target

    def __len__(self):
        return len(self.ids)
