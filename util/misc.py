"""
General purpose utility functions.

"""

# Utils
import logging
import os
import os.path
import shutil
import string

import cv2
from PIL import Image
import numpy as np
import torch

from sklearn.preprocessing import OneHotEncoder


def _prettyprint_logging_label(logging_label):
    """Format the logging label in a pretty manner.

    Parameters
    ----------
    logging_label : str
        The label used in logging

    Returns
    -------
    logging_label : str
        Correctly formatted logging label.

    """
    if len(logging_label) < 5:
        for i in range(5 - len(logging_label)):
            logging_label = logging_label + ' '
    return logging_label


class AverageMeter(object):
    """Computes and stores the average and current value

    From https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    """
    Computes the accuracy@K for the specified values of K

    From https://github.com/pytorch/examples/blob/master/imagenet/main.py

    Parameters
    ----------
    :param output:
        The output of the model

    :param target:
        The GT for the corresponding output

    :param topk:
        Top@k return value. It can be a tuple (1,5) and it return Top1 and Top5

    :return:
        Top@k accuracy
    """


def adjust_learning_rate(lr, optimizer, epoch, decay_lr_epochs):
    """Sets the learning rate to the initial LR decayed by 10 every N epochs.

    Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py

    Parameters
    ----------
    lr : float
        Learning rate.
    optimizer : torch.optim object
        The optimizer used for training the network.
    epoch : int
        Current training epoch.
    decay_lr_epochs : int
        Change the learning rate every N epochs.

    Returns
    -------
    None

    """
    import copy
    original_lr = copy.deepcopy(lr)
    lr = lr * (0.1 ** (epoch // decay_lr_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if original_lr != lr:
        logging.info('Learning rate decayed. New learning rate is: {}'.format(lr))
    return


def checkpoint(epoch, new_value, best_value, model, optimizer, log_dir,
               invert_best=False, checkpoint_all_epochs=False):
    """Saves the current training checkpoint and the best valued checkpoint to file.

    Parameters
    ----------
    epoch : int
        Current epoch, for logging purpose only.
    new_value : float
        Current value achieved by the model at this epoch.
        To be compared with 'best_value'.
    best_value : float
        Best value ever obtained (so the last checkpointed model).
        To be compared with 'new_value'.
    model : torch.nn.module object
        The model we are checkpointing, this can be saved on file if necessary.
    optimizer :
        The optimizer that is being used to train this model.
        It is necessary if we were to resume the training from a checkpoint.
    log_dir : str
        Output folder where to put the model.
    invert_best : bool
        Changes the scale such that smaller values are better than bigger values
        (useful when metric evaluted is error rate)
    checkpoint_all_epochs : bool
        If enabled, save checkpoint after every epoch.

    Returns
    -------
    best_value : float
        Best value ever obtained.

    """
    if invert_best:
        is_best = new_value < best_value
        best_value = min(new_value, best_value)
    else:
        is_best = new_value > best_value
        best_value = max(new_value, best_value)
    filename = os.path.join(log_dir, 'checkpoint.pth.tar')
    torch.save({
        'epoch': epoch + 1,
        'arch': str(type(model)),
        'state_dict': model.state_dict(),
        'best_value': best_value,
        'optimizer': optimizer.state_dict(),
    }, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.split(filename)[0], 'model_best.pth.tar'))
    # If enabled, save all checkpoints with epoch number.
    if checkpoint_all_epochs == True:
        shutil.move(filename, os.path.join(os.path.split(filename)[0], 'checkpoint_{}.pth.tar'.format(epoch)))
    return best_value


def to_capital_camel_case(s):
    """Converts a string to camel case.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        Input string `s` converted to camel case.

    """
    return s[0].capitalize() + string.capwords(s, sep='_').replace('_', '')[1:] if s else s


def get_all_files_in_folders_and_subfolders(root_dir=None):
    """Get all the files in a folder and sub-folders.

    Parameters
    ----------
    root_dir : str
        All files in this directory and it's sub-folders will be returned by this method.

    Returns
    -------
    paths : list of str
        List of paths to all files in this folder and it's subfolders.
    """
    paths = []
    for path, subdirs, files in os.walk(root_dir):
        for name in files:
            paths.append(os.path.join(path, name))
    return paths


def tensor_to_image(image):
    """
    Tries to reshape, convert and do operations necessary to bring the image
    in a format friendly to be saved and logged to Tensorboard by
    save_image_and_log_to_tensorboard()

    Parameters
    ----------
    image : ?
        Image to be converted

    Returns
    -------
    image : ndarray [W x H x C]
        Image, as format friendly to be saved and logged to Tensorboard.

    """
    # Check if the data is still a Variable()
    if 'variable' in str(type(image)):
        image = image.data

    # Check if the data is still on CUDA
    if 'cuda' in str(type(image)):
        image = image.cpu()

    # Check if the data is still on a Tensor
    if 'Tensor' in str(type(image)):
        image = image.numpy()
    assert ('ndarray' in str(type(image)))  # Its an ndarray

    # Check that it does not have anymore the 4th dimension (from the mini-batch)
    if len(image.shape) > 3:
        assert (len(image.shape) == 4)
        image = np.squeeze(image)
    assert (len(image.shape) == 3)  # 3D matrix (W x H x C)

    # Check that the last channel is of size 3 for RGB
    if image.shape[2] != 3:
        assert (image.shape[0] == 3)
        image = np.transpose(image, (1, 2, 0))
    assert (image.shape[2] == 3)  # Last channel is of size 3 for RGB

    # Check that the range is [0:255]
    image = (image - image.min()) / (image.max() - image.min())
    image = image * 255
    assert (image.min() >= 0)  # Data should be in range [0:255]

    return image

def save_image_and_log_to_tensorboard(writer=None, tag=None, image=None, global_step=None, segmentation=False):
    """Utility function to save image in the output folder and also log it to Tensorboard.

    Parameters
    ----------
    writer : tensorboardX.writer.SummaryWriter object
        The writer object for Tensorboard
    tag : str
        Name of the image.
    image : ndarray [W x H x C]
        Image to be saved and logged to Tensorboard.
    global_step : int
        Epoch/Mini-batch counter.

    Returns
    -------
    None

    """
    # Log image to Tensorboard
    writer.add_image(tag=tag, img_tensor=image, global_step=global_step)

    # Get output folder using the FileHandler from the logger.
    # (Assumes the file handler is the last one)
    output_folder = os.path.dirname(logging.getLogger().handlers[-1].baseFilename)

    if global_step is not None:
        dest_filename = os.path.join(output_folder, 'images', tag + '_{}.png'.format(global_step))
    else:
        dest_filename = os.path.join(output_folder, 'images', tag + '.png')

    if not os.path.exists(os.path.dirname(dest_filename)):
        os.makedirs(os.path.dirname(dest_filename))

    # Ensuring the data passed as parameter is healthy
    if segmentation:
        # Check that the last channel is of size 3 for RGB
        if image.shape[2] != 3:
            assert (image.shape[0] == 3)
            image = np.transpose(image, (1, 2, 0))
    else:
        image = tensor_to_image(image)

    # Write image to output folder
    cv2.imwrite(dest_filename, image)

    return

def save_image_and_log_to_tensorboard_segmentation(writer=None, tag=None, image=None, global_step=None, gt_img_path=None):
    """Utility function to save image in the output folder and also log it to Tensorboard.

    Parameters
    ----------
    writer : tensorboardX.writer.SummaryWriter object
        The writer object for Tensorboard
    tag : str
        Name of the image.
    image : ndarray [W x H x C]
        Image to be saved and logged to Tensorboard.
    global_step : int
        Epoch/Mini-batch counter.

    Returns
    -------
    None

    """
    #TODO pass this as argument
    int_val_to_class_name = {1: "background", 2: "comment", 4: "decoration", 6: "comment_decoration",
                             8: "maintext", 10: "maintext_comment", 12: "maintext_decoration",
                             14: "maintext_comment_decoration"}

    # 1. Create true output
    # Log image to Tensorboard
    #writer.add_image(tag=tag, img_tensor=image, global_step=global_step)

    # Get output folder using the FileHandler from the logger.
    # (Assumes the file handler is the last one)
    output_folder = os.path.dirname(logging.getLogger().handlers[-1].baseFilename)

    if global_step is not None:
        dest_filename = os.path.join(output_folder, 'images', tag + '_{}.png'.format(global_step))
    else:
        dest_filename = os.path.join(output_folder, 'images', tag + '.png')

    if not os.path.exists(os.path.dirname(dest_filename)):
        os.makedirs(os.path.dirname(dest_filename))

    cv2.imwrite(dest_filename, image)

    # 2. Make a more human readable output -> one colour per class
    # Write image to output folder
    tag_col = "coloured_" + tag

    # Get output folder using the FileHandler from the logger.
    # (Assumes the file handler is the last one)
    output_folder = os.path.dirname(logging.getLogger().handlers[-1].baseFilename)

    if global_step is not None:
        dest_filename = os.path.join(output_folder, 'images', tag_col + '_{}.png'.format(global_step))
    else:
        dest_filename = os.path.join(output_folder, 'images', tag_col + '.png')

    if not os.path.exists(os.path.dirname(dest_filename)):
        os.makedirs(os.path.dirname(dest_filename))

    img = np.copy(image)
    blue = img[:, :, 2]  # Extract just blue channel
    masks = {c: (blue == i) > 0 for i, c in int_val_to_class_name.items()}
    # Colours are in BGR
    class_col = {"background": (0, 0, 0), "maintext": (255, 255, 0), "comment": (0, 255, 255),
                 "decoration": (255, 0, 255),
                 "comment_decoration": (0, 125, 255), "maintext_comment": (0, 200, 0),
                 "maintext_decoration": (200, 0, 200),
                 "maintext_comment_decoration": (255, 255, 255)}

    for c, mask in masks.items():
        img[mask] = class_col[c]

    #writer.add_image(tag=tag_col, img_tensor=np.moveaxis(img, -1, 0), global_step=global_step)
    # Write image to output folder
    cv2.imwrite(dest_filename, img)

    # 3. Output image as described in https://github.com/DIVA-DIA/DIVA_Layout_Analysis_Evaluator
    # GREEN: Foreground predicted correctly rgb(80, 140, 30)
    # YELLOW: Foreground predicted - but the wrong class (e.g. Text instead of Comment) rgb(250, 230, 60)
    # BLACK: Background predicted correctly rgb(0, 0, 0)
    # RED: Background mis-predicted as Foreground rgb(240, 30, 20)
    # BLUE: Foreground mis-predicted as Background rgb(0, 240, 255)
    if gt_img_path:
        with open(gt_img_path, 'rb') as f:
            with Image.open(f) as img:
                ground_truth = np.array(img.convert('BGR'))

        img_la = np.copy(image)
        tag_la = "layout_analysis_" + tag
        # Get output folder using the FileHandler from the logger.
        # (Assumes the file handler is the last one)
        output_folder = os.path.dirname(logging.getLogger().handlers[-1].baseFilename)

        if global_step is not None:
            dest_filename = os.path.join(output_folder, 'images', tag_la + '_{}.png'.format(global_step))
        else:
            dest_filename = os.path.join(output_folder, 'images', tag_la + '.png')

        if not os.path.exists(os.path.dirname(dest_filename)):
            os.makedirs(os.path.dirname(dest_filename))

        out_blue = img_la[:, :, 0]  # Extract just blue channel
        gt_blue = ground_truth[:, :, 0]

        img_la = np.array([[_get_colour(out_blue[x, y], gt_blue[x, y]) for y in range(out_blue.shape[1])]
                           for x in range(out_blue.shape[0])])

        # Write image to output folder
        # TODO this does not work
        #writer.add_image(tag=tag_la, img_tensor=np.moveaxis(img_la, -1, 0), global_step=global_step)
        cv2.imwrite(dest_filename, img_la)

    return

def _get_colour(output, gt):
    # Colours are in BGR
    class_col = {"fg_correct": (30, 160, 70), "fg_wrong_class": (60, 255, 255), "bg_correct": (0, 0, 0),
                 "bg_as_fg": (20, 30, 240), "fg_as_bg": (255, 240, 0)}

    if output == gt and gt in [2, 4, 6, 8, 10, 12, 14]:
        return class_col["fg_correct"]
    elif output == gt and gt == 1:
        return class_col["bg_correct"]
    elif output != gt and output in [2, 4, 6, 8, 10, 12, 14] and gt in [2, 4, 6, 8, 10, 12, 14]:
        return class_col["fg_wrong_class"]
    elif output != gt and output == 1:
        return class_col["fg_as_bg"]
    elif output != gt and output in [2, 4, 6, 8, 10, 12, 14]:
        return class_col["bg_as_fg"]
    else:
        return (255, 255, 255)


def has_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py.

    Parameters
    ----------
    filename : string
        path to a file
    extensions : list
        extensions to match against
    Returns
    -------
    bool
        True if the filename ends with one of given extensions, false otherwise.
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


# functions added for HisDB classification
def int_to_one_hot(x, n_classes):
    """
    Read out class encoding from blue channel bit-encoding (1 to [0,0,0,1] -> length determined by the number of classes)

    Parameters
    ----------
    x: int
        (pixel value of Blue channel from RGB image)
    n_classes: int
        number of class labels
    Returns
    -------
    list
        (multi) one-hot encoded list for integer
    """
    s = '{0:0' + str(n_classes) + 'b}'
    return list(map(int, list(s.format(x))))


def gt_tensor_to_one_hot(gt_tensor):
    """
    Convert ground truth tensor to one-hot encoded matrix

    Parameters
    -------
    gt_tensor: float tensor from to_tensor()
        torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    Returns
    -------
    torch.LongTensor of size [#C x H x W]
        sparse one-hot encoded multi-class matrix, where #C is the number of classes
    """
    # TODO: ugly fix -> better to not normalize in the first place
    np_array =(gt_tensor*255).numpy().astype(np.uint8)
    im_np = np_array[2, :, :].astype(np.uint8)

    integer_encoded = np.array([i for i in range(8)])
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded).astype(np.int8)

    replace_dict = {k: v for k, v in zip([1, 2, 4, 6, 8, 10, 12, 14], onehot_encoded)}
    # create the one hot matrix
    one_hot_matrix = np.asanyarray(
        [[replace_dict[im_np[i, j]] for j in range(im_np.shape[1])] for i in range(im_np.shape[0])]).astype(np.uint8)

    return torch.LongTensor(one_hot_matrix.transpose((2, 0, 1)))


def multi_label_img_to_multi_hot(np_array):
    """
    TODO: There must be a faster way of doing this + ajust to correct input format (see gt_tensor_to_one_hot)
    Convert ground truth label image to multi-one-hot encoded matrix of size image height x image width x #classes

    Parameters
    -------
    np_array: numpy array
        RGB image [W x H x C]
    Returns
    -------
    numpy array of size [#C x W x H]
        sparse one-hot encoded multi-class matrix, where #C is the number of classes
    """
    im_np = np_array[:, :, 2].astype(np.int8)
    nb_classes = len(int_to_one_hot(im_np.max(), ''))

    class_dict = {x: int_to_one_hot(x, nb_classes) for x in np.unique(im_np)}
    # create the one hot matrix
    one_hot_matrix = np.asanyarray(
        [[class_dict[im_np[i, j]] for j in range(im_np.shape[1])] for i in range(im_np.shape[0])])

    return np.rollaxis(one_hot_matrix.astype(np.uint8), 2, 0)


def multi_one_hot_to_output(matrix):
    """
    This function converts the multi-one-hot encoded matrix to an image like it was provided in the ground truth

    Parameters
    -------
    tensor of size [#C x W x H]
        sparse one-hot encoded multi-class matrix, where #C is the number of classes
    Returns
    -------
    np_array: numpy array
        RGB image [C x W x H]
    """
    # TODO: fix input and output dims (see one_hot_to_output)
    # create RGB
    matrix = np.rollaxis(np.char.mod('%d', matrix.numpy()), 0, 3)
    zeros = (32 - matrix.shape[2]) * '0'
    B = np.array([[int('{}{}'.format(zeros, ''.join(matrix[i][j])), 2) for j in range(matrix.shape[1])] for i in
                  range(matrix.shape[0])])

    RGB = np.dstack((np.zeros(shape=(matrix.shape[0], matrix.shape[1], 2), dtype=np.int8), B))

    return RGB


def one_hot_to_np_bgr(matrix):
    """
    This function converts the one-hot encoded matrix to an image like it was provided in the ground truth

    Parameters
    -------
    np array of size [#C x H x W]
        sparse one-hot encoded multi-class matrix, where #C is the number of classes
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
                                                 (one_hot[:, :zero_mask.shape[1], :zero_mask.shape[2]] +
                                                  combined_one_hot[:, x1:x2, y1:y2]) / 2)

    return combined_one_hot















