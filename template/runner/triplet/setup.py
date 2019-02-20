# Utils
from __future__ import print_function

import logging
import os

# Torch
import torchvision.transforms as transforms
from template.runner.triplet.transforms import MultiCrop

import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

# DeepDIVA
from datasets.image_folder_triplet import load_dataset
from template.setup import _dataloaders_from_datasets, _load_mean_std_from_file


def setup_dataloaders(model_expected_input_size, dataset_folder, n_triplets, batch_size, workers, inmem, **kwargs):
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
                                             num_triplets=n_triplets)

    # Loads the analytics csv and extract mean and std
    mean, std = _load_mean_std_from_file(dataset_folder=dataset_folder, inmem=inmem, workers=workers, runner_class=kwargs['runner_class'])
    if kwargs['multi_crop'] is None:
        kwargs['multi_crop'] = 1

    # Set up dataset transforms
    logging.debug('Setting up dataset transforms')

    #standard transform
    standard_transform = transforms.Compose([
        #transforms.Resize(size=model_expected_input_size)
        transforms.RandomCrop(size=model_expected_input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    # logging.info("Transform is set to RandomCrop")

    multicrop_transform = transforms.Compose([
        MultiCrop(size=model_expected_input_size, n_crops=kwargs['multi_crop']),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(
            lambda items: torch.stack([transforms.Normalize(mean=mean, std=std)(item) for item in items])),
        # transforms.RandomCrop(model_expected_input_size),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)

    ])
    logging.info("Transform is set to MultiCrop")

    train_ds.transform = standard_transform
    val_ds.transform = multicrop_transform
    test_ds.transform = multicrop_transform

    train_loader, val_loader, test_loader = _dataloaders_from_datasets(batch_size=batch_size,
                                                                       train_ds=train_ds,
                                                                       val_ds=val_ds,
                                                                       test_ds=test_ds,
                                                                       workers=workers)
    return train_loader, val_loader, test_loader
