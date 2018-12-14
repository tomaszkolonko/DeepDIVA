"""
Load a dataset of images by specifying the folder where its located.
"""

# Utils
import logging
import sys
import math
from multiprocessing import Pool
import cv2
import numpy as np

# Torch related stuff
import torchvision
from PIL import Image, ImageFile

import torch.utils.data as data

import os
import os.path

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

from util.misc import get_all_files_in_folders_and_subfolders, has_extension


def load_dataset(dataset_folder, in_memory=False, workers=1):
    """
    Loads the dataset from file system and provides the dataset splits for train validation and test

    The dataset is expected to be in the following structure, where 'dataset_folder' has to point to
    the root of the three folder train/val/test.

    Example:

        dataset_folder = "~/../../data/cifar"

    which contains the splits sub-folders as follow:

        'dataset_folder'/train
        'dataset_folder'/val
        'dataset_folder'/test

    In each of the three splits (train, val, test) should have different classes in a separate folder
    with the class name. The file name can be arbitrary i.e. it does not have to be 0-* for classes 0
    of MNIST.

    Example:

        train/dog/whatever.png
        train/dog/you.png
        train/dog/like.png

        train/cat/123.png
        train/cat/nsdf3.png
        train/cat/asd932_.png

        train/"class_name"/*.png

    Parameters
    ----------
    dataset_folder : string
        Path to the dataset on the file System

    in_memory : boolean
        Load the whole dataset in memory. If False, only file names are stored and images are loaded
        on demand. This is slower than storing everything in memory.

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

    # If its requested online, delegate to torchvision.datasets.ImageFolder()
    if not in_memory:
        # Get an online dataset for each split
        train_ds = ImageFolder(train_dir)
        val_ds = ImageFolder(val_dir)
        test_ds = ImageFolder(test_dir)
        return train_ds, val_ds, test_ds


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

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

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.current_batch = 0

        self.test_set = "/test" in self.root
        if self.test_set:
            self.current_image = 0
            self.number_of_test_images = len(self.imgs)
            self.current_horizontal_crop = 0
            self.current_vertical_crop = 0
            self.exhaustive_crops_per_test_set = self.get_exhaustive_crops_per_test_set()

    def get_exhaustive_crops_per_test_set(self):
        total_exhaustive_crops_per_test_epoch = 0
        for i in range(len(self.imgs)):
            image_file = Image.open(self.imgs[i][0])

            total_horizontal_crops_per_image = math.ceil(image_file.height / 224)
            total_vertical_crops_per_image = math.ceil(image_file.width / 224)

            total_exhaustive_crops_per_test_epoch += total_horizontal_crops_per_image * total_vertical_crops_per_image
        return total_exhaustive_crops_per_test_epoch



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)

        # TODO: for later -> think how to change the minibatches to match only one image at a time
        # but only at test time.

        if self.test_set:
            # return crop by crop of an image until one image is through
            # keep track of the positions per image with the self.current_horizontal_crop...

            print("yolo")


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if "unit_tests" in path:
            return img, target, path
        else:
            return img, target




    def __len__(self):
        if self.test_set:
            return self.exhaustive_crops_per_test_set

        return len(self.imgs)