# Utils
import logging
import os
import numpy as np
import numbers


# TODO: from __future__ import print_function
import torch
import torch.nn as nn

import torch.optim as optim
import torchvision
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

# Torch
from template.runner.semantic_segmentation_hisdb.transform_library import transforms

# DeepDIVA
from datasets.image_folder_segmentation import load_dataset
from template.setup import _dataloaders_from_datasets, _load_mean_std_from_file


def set_up_dataloaders(model_expected_input_size, dataset_folder, batch_size, workers, inmem, **kwargs):
    # TODO: refactor into the image_folder_segmentation.py
    """
    Set up the dataloaders for the specified datasets.

    Parameters
    ----------
    model_expected_input_size : tuple
        Specify the height and width that the model expects.
    dataset_folder : string
        Path string that points to the three folder train/val/test. Example: ~/../../data/svhn
    n_triplets : int
        Number of triplets to generate for train/val/tes
    batch_size : int
        Number of datapoints to process at once
    workers : int
        Number of workers to use for the dataloaders
    inmem : boolean
        Flag : if False, the dataset is loaded in an online fashion i.e. only file names are stored
        and images are loaded on demand. This is slower than storing everything in memory.


    Returns
    -------
    train_loader : torch.utils.data.DataLoader
    val_loader : torch.utils.data.DataLoader
    test_loader : torch.utils.data.DataLoader
        Dataloaders for train, val and test.
    """

    # Recover dataset name
    dataset = os.path.basename(os.path.normpath(dataset_folder))
    logging.info('Loading {} from:{}'.format(dataset, dataset_folder))

    ###############################################################################################
    # Load the dataset splits as images
    train_ds, val_ds, test_ds = load_dataset(dataset_folder=dataset_folder,
                                             in_memory=inmem,
                                             workers=workers,
                                             **kwargs)

    # Set up dataset transforms
    logging.debug('Setting up dataset transforms')

    image_gt_transform = transforms.Compose([
        transforms.RandomTwinCrop(),
        transforms.ToTensorTwinImage()
    ])

    train_ds.transform = image_gt_transform
    val_ds.transform = image_gt_transform
    test_ds.transform = image_gt_transform

    train_loader, val_loader, test_loader = _dataloaders_from_datasets(batch_size=batch_size,
                                                                       train_ds=train_ds,
                                                                       val_ds=val_ds,
                                                                       test_ds=test_ds,
                                                                       workers=workers)
    return train_loader, val_loader, test_loader
