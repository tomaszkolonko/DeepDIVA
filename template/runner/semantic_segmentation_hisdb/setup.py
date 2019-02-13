# Utils
import logging
import os
from sklearn.preprocessing import OneHotEncoder


# TODO: from __future__ import print_function
import torch

import numpy as np

# Torch
from datasets.transform_library import transforms

# DeepDIVA
from datasets.image_folder_segmentation import load_dataset


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
                                             workers=workers, **dict(kwargs, gt_to_one_hot=gt_to_one_hot))

    # Set up dataset transforms
    logging.debug('Setting up dataset transforms')

    image_gt_transform = transforms.Compose([
        transforms.RandomTwinCrop(),
        transforms.ToTensorTwinImage()
    ])

    train_ds.transform = image_gt_transform
    val_ds.transform = image_gt_transform
    test_ds.transform = image_gt_transform

    # Setup dataloaders
    logging.debug('Setting up dataloaders (#workers for test set to 1)')
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               shuffle=True,
                                               batch_size=batch_size,
                                               num_workers=workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds,
                                             batch_size=batch_size,
                                             num_workers=workers,
                                             pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds,
                                              batch_size=batch_size,
                                              num_workers=1,
                                              pin_memory=True)

    return train_loader, val_loader, test_loader


def one_hot_to_np_bgr(matrix):
    """
    This function converts the one-hot encoded matrix to an image like it was provided in the ground truth

    Parameters
    -------
    np array of size [#C x H x W]
        sparse one-hot encoded matrix, where #C is the number of classes
    Returns
    -------
    numpy array of size [C x H x W] (BGR)
    """
    B = np.argmax(matrix, axis=0)
    class_to_B = {i: j for i, j in enumerate([1, 2, 4, 6, 8, 10, 12, 14])}

    masks = [B == old for old in class_to_B.keys()]

    for mask, (old, new) in zip(masks, class_to_B.items()):
        B = np.where(mask, new, B)

    bgr = np.dstack((B, np.zeros(shape=(B.shape[0], B.shape[1], 2), dtype=np.int8)))

    return bgr


def one_hot_to_full_output(one_hot, coordinates, combined_one_hot, output_dim):
    """
    This function combines the one-hot matrix of all the patches in one image to one large output matrix. Overlapping
    values are averaged.

    Parameters
    ----------
    output_dims: tuples [Htot x Wtot]
        dimension of the large image
    one_hot: numpy matrix of size [batch size x #C x H x W]
        a patch from the larger image
    coordinates: tuple
        top left coordinates of the patch within the larger image for all patches in a batch
    combined_one_hot: numpy matrix of size [#C x Htot x Wtot]
        one hot encoding of the full image
    Returns
    -------
    combined_one_hot: numpy matrix [#C x Htot x Wtot]
    """
    if len(combined_one_hot) == 0:
        combined_one_hot = np.zeros((one_hot.shape[0], *output_dim))

    x1, y1 = coordinates
    x2, y2 = (min(x1 + one_hot.shape[1], output_dim[0]), min(y1 + one_hot.shape[2], output_dim[1]))
    zero_mask = combined_one_hot[:, x1:x2, y1:y2] == 0
    # if still zero in combined_one_hot just insert value from crop, if there is a value average
    combined_one_hot[:, x1:x2, y1:y2] = np.where(zero_mask, one_hot[:, :zero_mask.shape[1], :zero_mask.shape[2]],
                                                 np.maximum(one_hot[:, :zero_mask.shape[1], :zero_mask.shape[2]],
                                                  combined_one_hot[:, x1:x2, y1:y2]))

    return combined_one_hot

def gt_to_one_hot(matrix, num_classes):
    """
    Convert ground truth tensor to one-hot encoded matrix

    Parameters
    -------
    matrix: float tensor from to_tensor() or numpy array
        shape (C x H x W) in the range [0.0, 1.0] or shape (H x W x C) BGR

    Returns
    -------
    torch.LongTensor of size [#C x H x W]
        sparse one-hot encoded multi-class matrix, where #C is the number of classes
    """
    # TODO: ugly fix -> better to not normalize in the first place
    if type(matrix).__module__ == np.__name__:
        im_np = matrix[:, :, 2].astype(np.uint8)
        border_mask = matrix[:, :, 0].astype(np.uint8) != 0
    else:
        np_array = (matrix * 255).numpy().astype(np.uint8)
        im_np = np_array[2, :, :].astype(np.uint8)
        border_mask = np_array[0, :, :].astype(np.uint8) != 0

    # ajust blue channel according to border pixel in red channel
    im_np[border_mask] = 1

    integer_encoded = np.array([i for i in range(num_classes)])
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded).astype(np.int8)

    np.place(im_np, im_np == 0, 1) # needed to deal with 0 fillers at the borders during testing (replace with background)
    replace_dict = {k: v for k, v in zip([1, 2, 4, 6, 8, 10, 12, 14], onehot_encoded)}

    # create the one hot matrix
    one_hot_matrix = np.asanyarray(
        [[replace_dict[im_np[i, j]] for j in range(im_np.shape[1])] for i in range(im_np.shape[0])]).astype(np.uint8)

    return torch.LongTensor(one_hot_matrix.transpose((2, 0, 1)))