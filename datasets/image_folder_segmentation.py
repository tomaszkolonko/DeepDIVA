"""
Load a dataset of historic documents by specifying the folder where its located.
"""

# Utils
import logging
import os
import sys
from typing import List

import torch.utils.data as data

import os.path

# Torch related stuff
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


def find_classes():
    classes = ["background", "foreground", "text", "decoration"]
    classes.sort()
    return classes


def make_dataset(directory):
    images = []
    directory = os.path.expanduser(directory)

    # TODO: fix this as soon it is working
    path_imgs = os.path.join(directory, "img")
    path_gts = os.path.join(directory, "gt")

    if not (os.path.isdir(path_imgs) or os.path.isdir(path_gts)):
        logging.error("folder img or gt not found in " + str(directory))

    for _, _, fnames in sorted(os.walk(path_imgs)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path_img = os.path.join(path_imgs, fname)
                fname_gt = fname[:-4] + ".png"
                path_gt = os.path.join(path_gts, fname_gt)
                item = (path_img, path_gt)
                images.append(item)

    return images


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
        logging.info("Something went wrong with the default_loader in image_folder_segmentation.py")
        sys.exit(-1)


def load_dataset(dataset_folder, args, in_memory=False, workers=1, testing=False):
    """
    Loads the dataset from file system and provides the dataset splits for train validation and test

    The dataset is expected to be in the following structure, where 'dataset_folder' has to point to
    the root of the three folder train/val/test.

    Example:

        dataset_folder = "~/../../data/histDoc"

    which contains the splits sub-folders as follow:

        'dataset_folder'/train
        'dataset_folder'/val
        'dataset_folder'/test

    In each of the three splits (train, val, test) there are two folders. One for the ground truth ("gt")
    and the other for the images ("img"). The ground truth image is of equal size and and encoded the
    following classes: background, foreground, text and decoration.

    Example:

        ../CB55/train/img/page23.png
        ../CB55/train/img/page231.png
        ../CB55/train/gt/page23.png
        ../CB55/train/gt/page231.png

        ../CB55/val/img
        ../CB55/val/gt
        ../CB55/test/img
        ../CB55/test/gt



    Parameters
    ----------
    dataset_folder : string
        Path to the dataset on the file System

    args : dict
        Dictionary of all the CLI arguments passed in

    in_memory : boolean
        Load the whole dataset in memory. If False, only file names are stored and images are loaded
        on demand. This is slower than storing everything in memory.

    workers: int
        Number of workers to use for the dataloaders
    
    testing: boolean
        Take another path if you are in testing phase

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
    logging.info("*** TZ_DEBUG: you are in the segmentation class... so far so good.")

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

    if testing:
        return train_dir, val_dir, test_dir

    # Get an online dataset for each split
    train_ds = ImageFolder(train_dir, args['pages_in_memory'], args['crops_per_page'])
    val_ds = ImageFolder(val_dir, args['pages_in_memory'], args['crops_per_page'])
    test_ds = ImageFolder(test_dir, args['pages_in_memory'], args['crops_per_page'])
    return train_ds, val_ds, test_ds


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/gt/xxx.png
        root/gt/xxy.png
        root/gt/xxz.png

        root/img/xxx.png
        root/img/xxy.png
        root/img/xxz.png

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
    def __init__(self, root, pages_in_memory=0, crops_per_page=0, crop_size=10, transform=None, target_transform=None,
                 loader=default_loader):
        classes = find_classes()
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.pages_in_memory = pages_in_memory
        self.crops_per_page = crops_per_page
        self.crop_size = crop_size

        self.counter_crops_per_page = 0
        self.images = [None] * pages_in_memory
        self.gt = [None] * pages_in_memory


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, groundtruth) where both are paths to the actual image.
        """

        print(self.images)

        # Think about moving this initialization to the constructor !!!
        if self.images[0] == None:
            self.initialize_ram()



        path_img, path_gt = self.imgs[index]
        img = self.loader(path_img)
        gt = self.loader(path_gt)


        # TODO: make transform crop out of two arguments... controlling through variable N
        # TODO: take N crops out of the same image.... you also should have a threshold
        # value and you will use th same images for as long as you don't reach the threshold....
        # return crops of img and return crops on gt of the one-hot encoder from linda.
        while(self.counter_crops_per_page <= self.crops_per_page):
            if self.transform is not None:
                img, gt = self.transform(img, gt, self.crop_size)
                self.counter_crops_per_page = self.counter_crops_per_page + 1
                return img, gt
        if(self.counter_crops_per_page > self.crops_per_page):
            self.update_ram()
            self.counter_crops_per_page = 0

        #if self.transform is not None:
        #    img = self.transform(img, gt)
        #if self.target_transform is not None:
        #    gt = self.transform(gt)

    def initialize_ram(self):
        for i in range(0, self.pages_in_memory):
            temp_image, temp_gt = self.imgs[i]

            self.images[i] = self.loader(temp_image)
            self.gt[i] = self.loader(temp_gt)

    def update_ram(self):

    def __len__(self):
        return len(self.imgs)