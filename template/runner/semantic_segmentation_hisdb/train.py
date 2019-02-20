# Utils
import logging
import time
import numpy as np
from PIL import Image
import os

# Torch related stuff
import torch
from tqdm import tqdm

# DeepDIVA
from util.misc import AverageMeter, _prettyprint_logging_label, save_image_and_log_to_tensorboard, \
    save_image_and_log_to_tensorboard_segmentation
from .setup import one_hot_to_np_bgr, gt_to_one_hot
from util.evaluation.metrics.accuracy import accuracy_segmentation

def train(train_loader, model, criterion, optimizer, writer, epoch, class_names, no_cuda=False, log_interval=25,
          **kwargs):
    """
    Training routine

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        The dataloader of the train set.
    model : torch.nn.module
        The network model being used.
    criterion : torch.nn.loss
        The loss function used to compute the loss of the model.
    optimizer : torch.optim
        The optimizer used to perform the weight update.
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.
    epoch : int
        Number of the epoch (for logging purposes).
    no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.
    log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    Returns
    ----------
    meanIU.avg : float
        meanIU of the model of the evaluated split
    """
    multi_run = kwargs['run'] if 'run' in kwargs else None
    num_classes = len(class_names)

    # Instantiate the counters
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    meanIU = AverageMeter()
    data_time = AverageMeter()

    # Switch to train mode (turn on dropout & stuff)
    model.train()

    # Iterate over whole training set
    end = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), unit='batch', ncols=150, leave=False)
    for batch_idx, (input, target_argmax) in pbar:
        # convert 3D one-hot encoded matrix to 2D matrix with class numbers (for CrossEntropy())
        target_argmax = torch.LongTensor(np.array([np.argmax(a, axis=0) for a in target_argmax.numpy()]))

        # Measure data loading time
        data_time.update(time.time() - end)

        # Moving data to GPU
        if not no_cuda:
            input = input.cuda(async=True)
            target_argmax = target_argmax.cuda(async=True)

        # Convert the input and its labels to Torch Variables
        input_var = torch.autograd.Variable(input)
        target_var_argmax = torch.autograd.Variable(target_argmax)

        mean_iu, loss = train_one_mini_batch(model, criterion, optimizer, input_var, target_var_argmax, loss_meter, meanIU, num_classes)

        # Add loss and accuracy to Tensorboard
        if multi_run is None:
            writer.add_scalar('train/mb_loss', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/mb_meanIU', mean_iu, epoch * len(train_loader) + batch_idx)
        else:
            writer.add_scalar('train/mb_loss_{}'.format(multi_run), loss.item(),
                              epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/mb_meanIU_{}'.format(multi_run), mean_iu,
                              epoch * len(train_loader) + batch_idx)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log to console
        if batch_idx % log_interval == 0:
            pbar.set_description('train epoch [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(train_loader)))

            pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                             Loss='{loss.avg:.4f}\t'.format(loss=loss_meter),
                             meanIU='{meanIU.avg:.3f}\t'.format(meanIU=meanIU),
                             Data='{data_time.avg:.3f}\t'.format(data_time=data_time))

    # Logging the epoch-wise accuracy
    if multi_run is None:
        writer.add_scalar('train/meanIU', meanIU.avg, epoch)
    else:
        writer.add_scalar('train/meanIU{}'.format(multi_run), meanIU.avg, epoch)

    logging.debug('Train epoch[{}]: '
                 'MeanIU={meanIU.avg:.3f}\t'
                 'Loss={loss.avg:.4f}\t'
                 'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                  .format(epoch, batch_time=batch_time, data_time=data_time, loss=loss_meter, meanIU=meanIU))

    # logging.info(_prettyprint_logging_label("train") +
    #              ' epoch[{}]: '
    #              'MeanIU={meanIU.avg:.3f}\t'
    #              'Loss={loss.avg:.4f}\t'
    #              'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
    #              .format(epoch, batch_time=batch_time, data_time=data_time, loss=loss_meter, meanIU=meanIU))

    return meanIU.avg


def train_one_mini_batch(model, criterion, optimizer, input_var, target_var_argmax, loss_meter, meanIU_meter, num_classes):
    """
    This routing train the model passed as parameter for one mini-batch

    Parameters
    ----------
    num_classes:

    model : torch.nn.module
        The network model being used.
    criterion : torch.nn.loss
        The loss function used to compute the loss of the model.
    optimizer : torch.optim
        The optimizer used to perform the weight update.
    input_var : torch.autograd.Variable
        The input data for the mini-batch
    target_var_argmax : torch.autograd.Variable
        The target data (labels) for the mini-batch
    loss_meter : AverageMeter
        Tracker for the overall loss
    meanIU_meter : AverageMeter
        Tracker for the overall meanIU

    Returns
    -------
    acc : float
        Accuracy for this mini-batch
    loss : float
        Loss for this mini-batch
    """
    # Compute output
    output = model(input_var)

    # Compute and record the loss
    loss = criterion(output, target_var_argmax)
    loss_meter.update(loss.item(), len(input_var))

    output_argmax = np.array([np.argmax(o, axis=0) for o in output.data.cpu().numpy()])
    target_argmax = target_var_argmax.data.cpu().numpy()

    # Compute and record the accuracy
    acc, acc_cls, mean_iu, fwavacc = accuracy_segmentation(target_argmax, output_argmax, num_classes)
    meanIU_meter.update(mean_iu, input_var.size(0))

    # Reset gradient
    optimizer.zero_grad()
    # Compute gradients
    loss.backward()
    # Perform a step by updating the weights
    optimizer.step()

    # return acc, loss
    return mean_iu, loss

def _save_test_img_output(img_to_save, one_hot, multi_run, dataset_folder, logging_label, writer, **kwargs):
    """
    Helper function to save the output during testing

    Parameters
    ----------
    meanIU.avg : float
        MeanIU of the model of the evaluated split
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.
    epoch : int
        Number of the epoch (for logging purposes)
    img_to_save: str
        name of the image that is saved
    one_hot: numpy array
        one hot encoded output of the network for the whole image
    dataset_folder: str
        path to the dataset folder
    """

    logging.info("image {}. Saving output...".format(img_to_save))
    np_bgr = one_hot_to_np_bgr(one_hot, **kwargs)
    # add full image to predictions
    pred = np.argmax(one_hot, axis=0)
    # open full ground truth image
    gt_img_path = os.path.join(dataset_folder, logging_label, "gt", img_to_save)
    with open(gt_img_path, 'rb') as f:
        with Image.open(f) as img:
            ground_truth = np.array(img.convert('RGB'))
            # ajust blue channel according to border pixel in red channel
            border_mask = ground_truth[:, :, 0].astype(np.uint8) != 0
            ground_truth[:, :, 2][border_mask] = 1

            # ground_truth_argmax = functional.to_tensor(ground_truth)
    target = np.argmax(gt_to_one_hot(ground_truth).numpy(), axis=0)

    # TODO: also save input and gt image?
    if multi_run is None:
        save_image_and_log_to_tensorboard_segmentation(writer, tag=logging_label + '/output_{}'.format(img_to_save),
                                                       image=np_bgr,
                                                       gt_image=ground_truth[:, :, ::-1])  # ground_truth[:, :, ::-1] convert image to BGR
    else:
        save_image_and_log_to_tensorboard_segmentation(writer, tag=logging_label + '/output_{}_{}'.format(multi_run,
                                                                                                          img_to_save),
                                                       image=np_bgr,
                                                       gt_image=ground_truth[:, :, ::-1])  # ground_truth[:, :, ::-1] convert image to BGR

    return

# def save_image_and_log_to_tensorboard_segmentation(writer=None, tag=None, image=None, global_step=None, gt_image=[]):
#     """Utility function to save image in the output folder and also log it to Tensorboard.
#     ALL IMAGES ARE IN BGR BECAUSE OF CV3.IMWRITE()!!
#
#     Parameters
#     ----------
#     writer : tensorboardX.writer.SummaryWriter object
#         The writer object for Tensorboard
#     tag : str
#         Name of the image.
#     image : ndarray [W x H x C]
#         Image to be saved and logged to Tensorboard.
#     global_step : int
#         Epoch/Mini-batch counter.
#
#     Returns
#     -------
#     None
#
#     """
#     #TODO pass this as argument
#     int_val_to_class_name = {1: "background", 2: "comment", 4: "decoration", 6: "comment_decoration",
#                              8: "maintext", 10: "maintext_comment", 12: "maintext_decoration",
#                              14: "maintext_comment_decoration"}
#
#     # 1. Create true output
#     # Log image to Tensorboard
#     #writer.add_image(tag=tag, img_tensor=image, global_step=global_step)
#
#     # Get output folder using the FileHandler from the logger.
#     # (Assumes the file handler is the last one)
#     output_folder = os.path.dirname(logging.getLogger().handlers[-1].baseFilename)
#
#     if global_step is not None:
#         dest_filename = os.path.join(output_folder, 'images', tag + '_{}'.format(global_step))
#     else:
#         dest_filename = os.path.join(output_folder, 'images', tag)
#
#     if not os.path.exists(os.path.dirname(dest_filename)):
#         os.makedirs(os.path.dirname(dest_filename))
#
#     cv2.imwrite(dest_filename, image)
#
#     # 2. Make a more human readable output -> one colour per class
#     # Write image to output folder
#     tag_col = "coloured_" + tag
#
#     # Get output folder using the FileHandler from the logger.
#     # (Assumes the file handler is the last one)
#     output_folder = os.path.dirname(logging.getLogger().handlers[-1].baseFilename)
#
#     if global_step is not None:
#         dest_filename = os.path.join(output_folder, 'images', tag_col + '_{}'.format(global_step))
#     else:
#         dest_filename = os.path.join(output_folder, 'images', tag_col)
#
#     if not os.path.exists(os.path.dirname(dest_filename)):
#         os.makedirs(os.path.dirname(dest_filename))
#
#     img = np.copy(image)
#     blue = image[:, :, 0]  # Extract just blue channel
#     masks = {c: (blue == i) > 0 for i, c in int_val_to_class_name.items()}
#     # Colours are in BGR
#     class_col = {"background": (0, 0, 0), "maintext": (255, 255, 0), "comment": (0, 255, 255),
#                  "decoration": (255, 0, 255),
#                  "comment_decoration": (0, 125, 255), "maintext_comment": (0, 200, 0),
#                  "maintext_decoration": (200, 0, 200),
#                  "maintext_comment_decoration": (255, 255, 255)}
#
#     for c, mask in masks.items():
#         img[mask] = class_col[c]
#
#     #writer.add_image(tag=tag_col, img_tensor=np.moveaxis(img, -1, 0), global_step=global_step)
#     # Write image to output folder
#     cv2.imwrite(dest_filename, img)
#
#     # 3. Output image as described in https://github.com/DIVA-DIA/DIVA_Layout_Analysis_Evaluator
#     # GREEN: Foreground predicted correctly rgb(80, 140, 30)
#     # YELLOW: Foreground predicted - but the wrong class (e.g. Text instead of Comment) rgb(250, 230, 60)
#     # BLACK: Background predicted correctly rgb(0, 0, 0)
#     # RED: Background mis-predicted as Foreground rgb(240, 30, 20)
#     # BLUE: Foreground mis-predicted as Background rgb(0, 240, 255)
#     if len(gt_image) != 0:
#         class_col = {"fg_correct": (30, 160, 70), "fg_wrong_class": (60, 255, 255), "bg_correct": (0, 0, 0),
#                      "bg_as_fg": (20, 30, 240), "fg_as_bg": (255, 240, 0)}
#
#         img_la = np.copy(image)
#         tag_la = "layout_analysis_" + tag
#         # Get output folder using the FileHandler from the logger.
#         # (Assumes the file handler is the last one)
#         output_folder = os.path.dirname(logging.getLogger().handlers[-1].baseFilename)
#
#         if global_step is not None:
#             dest_filename = os.path.join(output_folder, 'images', tag_la + '_{}'.format(global_step))
#         else:
#             dest_filename = os.path.join(output_folder, 'images', tag_la)
#
#         if not os.path.exists(os.path.dirname(dest_filename)):
#             os.makedirs(os.path.dirname(dest_filename))
#
#         out_blue = image[:, :, 0]  # Extract just blue channel
#         gt_blue = gt_image[:, :, 0]
#
#         masks = {c: _get_mask(c, out_blue, gt_blue) for c in class_col.keys()}
#
#         for c, mask in masks.items():
#             img_la[mask] = class_col[c]
#
#         # img_la = np.array([[_get_colour(out_blue[x, y], gt_blue[x, y]) for y in range(out_blue.shape[1])]
#         #                    for x in range(out_blue.shape[0])])
#         # Write image to output folder
#         # TODO this does not work
#         #writer.add_image(tag=tag_la, img_tensor=np.moveaxis(img_la, -1, 0), global_step=global_step)
#         cv2.imwrite(dest_filename, img_la)
#
#     return
