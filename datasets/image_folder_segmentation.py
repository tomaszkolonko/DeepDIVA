"""
Load a dataset of historic documents by specifying the folder where its located.
"""

# Utils
import logging
import os
import sys

from datasets.image_folder_segmentation_parser import ImageFolder

from multiprocessing import Pool
import cv2
import numpy as np

# Torch related stuff
import torch.utils.data as data
import torchvision
from PIL import Image

from util.misc import get_all_files_in_folders_and_subfolders, has_extension


def load_dataset(dataset_folder, in_memory=False, workers=1):
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
    TODO: check if correct and maybe say something about the encoding

    Example:

        train/gt/mustBeSameName_1.png
        train/gt/mustBeSameName_2.png
        train/gt/whatever_34.png

        train/img/mustBeSameName_1.png
        train/img/mustBeSameName_2.png
        train/img/whatever_34.png

    Parameters
    ----------
    dataset_folder : string
        Path to the dataset on the file System

    TODO: check if we want to give this option or not
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

    # Get an online dataset for each split
    train_ds = ImageFolder(train_dir)
    val_ds = ImageFolder(val_dir)
    test_ds = ImageFolder(test_dir)
    return train_ds, val_ds, test_ds


