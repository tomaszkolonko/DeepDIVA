# Utils
import logging
import os
import sys

# TODO: from __future__ import print_function
import torchvision

# Torch

# DeepDIVA
from datasets.transform_library import transforms, functional
from template.setup import _dataloaders_from_datasets
from datasets.coco_detection import CocoDetection


def set_up_dataloaders(dataset_folder, batch_size, workers, **kwargs):
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
    train_loader : pycocotools.coco.COCO
    val_loader : pycocotools.coco.COCO
    test_loader : pycocotools.coco.COCO
        Dataloaders for train, val and test.
    """

    # Recover dataset name
    dataset = os.path.basename(os.path.normpath(dataset_folder))
    logging.info('Loading {} from:{}'.format(dataset, dataset_folder))

    # Get the splits folders
    train_dir = os.path.join(dataset_folder, 'train')
    val_dir = os.path.join(dataset_folder, 'val')
    test_dir = os.path.join(dataset_folder, 'test')
    gt_dir = os.path.join(dataset_folder, 'annotations')

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

    ###############################################################################################
    # Set up dataset transforms
    logging.debug('Setting up dataset transforms')

    # input_transform = transforms.Normalize()
    input_transform = functional.to_tensor
    target_transform = None

    # Setup dataloaders
    logging.debug('Setting up dataloaders')
    train_ds = CocoDetection(train_dir, jsons['train'])
    name_onehotindex = train_ds.name_onehotindex
    category_id_name = train_ds.category_id_name

    val_ds = CocoDetection(val_dir, jsons['val'], name_onehotindex=name_onehotindex, category_id_name=category_id_name)

    test_ds = CocoDetection(test_dir, jsons['test'], name_onehotindex=name_onehotindex, category_id_name=category_id_name)

    train_loader, val_loader, test_loader = _dataloaders_from_datasets(batch_size, train_ds, val_ds, test_ds, workers)

    return train_loader, val_loader, test_loader, name_onehotindex, category_id_name
