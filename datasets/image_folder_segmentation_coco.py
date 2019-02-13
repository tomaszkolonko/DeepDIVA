"""
Load the coco dataset of images by specifying the folder where its located.
"""

# Utils
import logging
import os
import sys
import math
import os.path
import numpy as np

from datasets.transform_library import transforms, functional

# Torch related stuff
import torch.utils.data as data
import torchvision

from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'PIL':
        return pil_loader(path)
    else:
        logging.info("Something went wrong with the default_loader in image_folder_segmentation_hisdb.py")
        sys.exit(-1)


def load_dataset(dataset_folder, in_memory=False, workers=1, **kwargs):
    """
    Loads the dataset from file system and provides the dataset splits for train validation and test

    The dataset is expected to be in the following structure, where 'dataset_folder' has to point to
    the root of the three folder train/val/test.

    Example:

        dataset_folder = "~/../../data/coco"

    which contains the splits sub-folders as follow:

        'dataset_folder'/train
        'dataset_folder'/val
        'dataset_folder'/test

    The ground truth is saved in json format and has to be located in the annotation subfolder

        ../'dataset_folder'/annotations/instances_train2017.json

    Parameters
    ----------
    dataset_folder : string
        Path to the dataset on the file System

    args : dict
        Dictionary of all the CLI arguments passed in

    workers: int
        Number of workers to use for the dataloaders

    Returns
    -------
    train_ds : data.Dataset

    val_ds : data.Dataset

    test_ds : data.Dataset
        Train, validation and test splits
    """
    # Get the splits folders
    train_dir = os.path.join(dataset_folder, 'train')
    val_dir = os.path.join(dataset_folder, 'val')
    test_dir = os.path.join(dataset_folder, 'test')
    gt_dir = os.path.join(dataset_folder, 'annotations')

    if in_memory:
        logging.error("With segmentation you don't have the option to put everything into memory")
        sys.exit(-1)

    # Sanity check on the splits folders
    if not os.path.isdir(train_dir):
        logging.error("Train folder not found in the dataset_folder=" + dataset_folder)
        sys.exit(-1)
    if not os.path.isdir(val_dir):
        logging.error("Val folder not found in the dataset_folder=" + dataset_folder)
        sys.exit(-1)
    if not os.path.isdir(test_dir):
        logging.error("Test folder not found in the dataset_folder=" + dataset_folder)
        sys.exit(-1)
    if not os.path.isdir(gt_dir):
        logging.error("Annotations folder not found in the dataset_folder=" + dataset_folder)
        sys.exit(-1)

    jsons = {}
    for s in ['train', 'val', 'test']:
        try:
            jsons[s] = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if
                        'instances_{}'.format(s) in f or 'info_{}'.format(s) in f][0]

        except FileNotFoundError:
            logging.error("Annotation JSON not found for {} set" + s)
            sys.exit(-1)

    # Get an online dataset for each split
    train_ds = ImageFolder(train_dir, jsons['train'], **kwargs)
    val_ds = ImageFolder(val_dir, jsons['val'], **kwargs)
    test_ds = ImageFolder(test_dir, jsons['test'], **kwargs)
    return train_ds, val_ds, test_ds


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    # TODO: transform and target_transform could be the correct places for your cropping
    def __init__(self, imgs_path, annotation_json, transform=None, target_transform=None,
                 loader=default_loader, **kwargs):
        self.dataset = torchvision.datasets.CocoDetection(imgs_path, annotation_json, transform=transform,
                                                          target_transform=target_transform)

        self.classes = [d['name'] for d in self.dataset.coco.dataset['categories']]
        self.imgs_path = imgs_path
        self.num_classes = len(self.classes)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, gt)
                gt is one-hot encoded
        """
        # TODO
        # convert PIL image to BGR, convert polygons of annotation to one-hot encoded matrix



    def __len__(self):
        """
        This function returns the length of an epoch so the dataloader knows when to stop
        :return:
        """
        return len(self.dataset)
