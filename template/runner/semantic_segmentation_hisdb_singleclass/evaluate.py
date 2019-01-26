# Utils
import logging
import time
import warnings
import os
import numpy as np
from PIL import Image
import cv2


# Torch related stuff
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# DeepDIVA
from util.misc import AverageMeter, _prettyprint_logging_label, save_image_and_log_to_tensorboard
from util.visualization.confusion_matrix_heatmap import make_heatmap
from util.evaluation.metrics.accuracy import accuracy_segmentation
from template.setup import _load_class_frequencies_weights_from_file
from .setup import one_hot_to_np_bgr, one_hot_to_full_output, gt_to_one_hot

def validate(data_loader, model, criterion, writer, epoch, class_names, dataset_folder, inmem, workers, runner_class, no_cuda=False, log_interval=10, **kwargs):
    """
    The evaluation routine

    Parameters
    ----------
    class_names : list
        Contains the class names
    data_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set
    model : torch.nn.module
        The network model being used
    criterion: torch.nn.loss
        The loss function used to compute the loss of the model
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.
    epoch : int
        Number of the epoch (for logging purposes)
    no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.
    log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    Returns
    -------
    meanIU.avg : float
        MeanIU of the model of the evaluated split
    """
    logging_label = "val"

    num_classes = len(class_names)
    multi_run = kwargs['run'] if 'run' in kwargs else None

    # Instantiate the counters
    batch_time = AverageMeter()
    losses = AverageMeter()
    meanIU = AverageMeter()
    data_time = AverageMeter()

    # Switch to evaluate mode (turn off dropout & such )
    model.eval()

    # Iterate over whole evaluation set
    end = time.time()

    # Empty lists to store the predictions and target values
    preds = []
    targets = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), unit='batch', ncols=150, leave=False)
    for batch_idx, (input, target) in pbar:
        # convert 3D one-hot encoded matrix to 2D matrix with class numbers (for CrossEntropy())
        target_argmax = torch.LongTensor(np.array([np.argmax(a, axis=0) for a in target.numpy()]))

        # Measure data loading time
        data_time.update(time.time() - end)

        # Moving data to GPU
        if not no_cuda:
            input = input.cuda(async=True)
            target = target.cuda(async=True)
            target_argmax = target_argmax.cuda(async=True)

        # Convert the input and its labels to Torch Variables
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_argmax_var = torch.autograd.Variable(target_argmax)

        # Compute output
        output = model(input_var)
        output_argmax = np.array([np.argmax(o, axis=0) for o in output.data.cpu().numpy()])

        # Compute and record the loss
        loss = criterion(output, target_argmax_var)
        losses.update(loss.data[0], input.size(0))

        # Compute and record the accuracy TODO check with Vinay & Michele if correct
        acc, acc_cls, mean_iu, fwavacc = accuracy_segmentation(target_argmax.cpu().numpy(), output_argmax, num_classes)
        meanIU.update(mean_iu, input.size(0))

        # Get the predictions
        _ = [preds.append(item) for item in output_argmax]
        _ = [targets.append(item) for item in target_argmax.cpu().numpy()]

        # Add loss and accuracy to Tensorboard
        if multi_run is None:
            writer.add_scalar(logging_label + '/mb_loss', loss.data[0], epoch * len(data_loader) + batch_idx)
            writer.add_scalar(logging_label + '/mb_meanIU', mean_iu, epoch * len(data_loader) + batch_idx)
        else:
            writer.add_scalar(logging_label + '/mb_loss_{}'.format(multi_run), loss.data[0],
                              epoch * len(data_loader) + batch_idx)
            writer.add_scalar(logging_label + '/mb_meanIU_{}'.format(multi_run), mean_iu,
                               epoch * len(data_loader) + batch_idx)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % log_interval == 0:
            pbar.set_description(logging_label +
                                 ' epoch [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(data_loader)))

            pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                             Loss='{loss.avg:.4f}\t'.format(loss=losses),
                             meanIU='{meanIU.avg:.3f}\t'.format(meanIU=meanIU),
                             Data='{data_time.avg:.3f}\t'.format(data_time=data_time))

        # for i, (inp, outp) in zip(output.data.cpu().numpy(), output.data.cpu().numpy()):
        #     save_image_and_log_to_tensorboard_segmentation(writer, tag=logging_label + '/output_{}_{}'.format(batch_idx, i),
        #                                                    image=outp,
        #                                                    gt_image=inp[:, :, ::-1])  # ground_truth[:, :, ::-1] convert image to BGR

    # Make a confusion matrix
    try:
        # targets_flat = np.array(targets).flatten()
        # preds_flat = np.array(preds).flatten()
        # calculate confusion matrices
        cm = confusion_matrix(y_true=np.array(targets).flatten(), y_pred=np.array(preds).flatten(), labels=[i for i in range(num_classes)])
        confusion_matrix_heatmap = make_heatmap(cm, class_names)

        # load the weights
        # weights = _load_class_frequencies_weights_from_file(dataset_folder, inmem, workers, runner_class)
        # sample_weight = [weights[i] for i in np.array(targets).flatten()]
        # cm_w = confusion_matrix(y_true=np.array(targets).flatten(), y_pred=np.array(preds).flatten(), labels=[i for i in range(num_classes)],
        #                         sample_weight=[weights[i] for i in np.array(targets).flatten()])
        # confusion_matrix_heatmap_w = make_heatmap(np.round(cm_w*100).astype(np.int), class_names)
    except ValueError:
        logging.warning('Confusion Matrix did not work as expected')
        confusion_matrix_heatmap = np.zeros((10, 10, 3))
        # confusion_matrix_heatmap_w = confusion_matrix_heatmap

    # Logging the epoch-wise accuracy and saving the confusion matrix
    if multi_run is None:
        writer.add_scalar(logging_label + '/meanIU', meanIU.avg, epoch)
        save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix',
                                          image=confusion_matrix_heatmap, global_step=epoch)
        # save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix_weighted',
        #                                   image=confusion_matrix_heatmap_w, global_step=epoch)
    else:
        writer.add_scalar(logging_label + '/meanIU_{}'.format(multi_run), meanIU.avg, epoch)
        save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix_{}'.format(multi_run),
                                          image = confusion_matrix_heatmap, global_step = epoch)
        # save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix_weighted{}'.format(multi_run),
        #                                   image=confusion_matrix_heatmap_w, global_step=epoch)


    logging.info(_prettyprint_logging_label(logging_label) +
                 ' epoch[{}]: '
                 'MeanIU={meanIU.avg:.3f}\t'
                 'Loss={loss.avg:.4f}\t'
                 'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                 .format(epoch, batch_time=batch_time, data_time=data_time, loss=losses, meanIU=meanIU))

    # # Generate a classification report for each epoch
    # _log_classification_report(data_loader, epoch, preds, targets, writer)

    return meanIU.avg


def test(data_loader, model, criterion, writer, epoch, class_names, dataset_folder, inmem, workers, runner_class, use_boundary_pixel, no_cuda=False, log_interval=10, **kwargs):
    """
    The evaluation routine

    Parameters
    ----------
    class_names : list
        Contains the class names
    data_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set
    model : torch.nn.module
        The network model being used
    criterion: torch.nn.loss
        The loss function used to compute the loss of the model
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.
    epoch : int
        Number of the epoch (for logging purposes)
    no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.
    log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    Returns
    -------
    meanIU.avg : float
        MeanIU of the model of the evaluated split
    """
    logging_label = "test"

    num_classes = len(class_names)
    multi_run = kwargs['run'] if 'run' in kwargs else None

    # Instantiate the counters
    batch_time = AverageMeter()
    losses = AverageMeter()
    meanIU = AverageMeter()
    data_time = AverageMeter()

    # Switch to evaluate mode (turn off dropout & such )
    model.eval()

    # Iterate over whole evaluation set
    end = time.time()

    # Empty lists to store the predictions and target values
    preds = []
    targets = []

    # needed for test phase output generation
    combined_one_hots = {}
    current_img_names = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), unit='batch', ncols=150, leave=False)
    for batch_idx, (input, target) in pbar:
        input, orig_img_shape, top_left_coordinates, test_img_names = input
        orig_img_shape = (orig_img_shape[0][0], orig_img_shape[1][0])

        # if not all('' == s or s.isspace() for s in test_img_names):
        #     print(test_img_names)

        # convert 3D one-hot encoded matrix to 2D matrix with class numbers (for CrossEntropy())
        target_argmax = torch.LongTensor(np.array([np.argmax(a, axis=0) for a in target.numpy()]))

        # Measure data loading time
        data_time.update(time.time() - end)

        # Moving data to GPU
        if not no_cuda:
            input = input.cuda(async=True)
            target = target.cuda(async=True)
            target_argmax = target_argmax.cuda(async=True)

        # Convert the input and its labels to Torch Variables
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_argmax_var = torch.autograd.Variable(target_argmax)

        # Compute output
        output = model(input_var)
        output_argmax = np.array([np.argmax(o, axis=0) for o in output.data.cpu().numpy()])

        # Compute and record the loss
        loss = criterion(output, target_argmax_var)
        losses.update(loss.data[0], input.size(0))

        # Compute and record the batch meanIU TODO check with Vinay & Michele if correct

        acc_batch, acc_cls_batch, mean_iu_batch, fwavacc_batch = accuracy_segmentation(target_argmax.cpu().numpy(), output_argmax, num_classes)
        #meanIU.update(mean_iu, input.size(0))

        # Add loss and accuracy to Tensorboard
        if multi_run is None:
            writer.add_scalar(logging_label + '/mb_loss', loss.data[0], epoch * len(data_loader) + batch_idx)
            writer.add_scalar(logging_label + '/mb_meanIU', mean_iu_batch, epoch * len(data_loader) + batch_idx)
        else:
            writer.add_scalar(logging_label + '/mb_loss_{}'.format(multi_run), loss.data[0],
                              epoch * len(data_loader) + batch_idx)
            writer.add_scalar(logging_label + '/mb_meanIU_{}'.format(multi_run), mean_iu_batch,
                               epoch * len(data_loader) + batch_idx)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % log_interval == 0:
            pbar.set_description(logging_label +
                                 ' epoch [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(data_loader)))

            pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                             Loss='{loss.avg:.4f}\t'.format(loss=losses),
                             meanIU='{meanIU.avg:.3f}\t'.format(meanIU=meanIU),
                             Data='{data_time.avg:.3f}\t'.format(data_time=data_time))

        # Output needs to be patched together to form the complete output of the full image
        # patches are returned as a sliding window over the full image, overlapping sections are averaged
        one_hots = output.data.cpu().numpy()
        for one_hot, x, y, img_name in zip(one_hots, top_left_coordinates[0].numpy(), top_left_coordinates[1].numpy(), test_img_names):
            # check if we are already working on the image passed in img_name
            if img_name not in current_img_names:
                current_img_names.append(img_name)
                combined_one_hots[img_name] = []

            # we are only working on two images
            if len(current_img_names) < 3:
                # on the same image / first iteration
                combined_one_hots[img_name] = one_hot_to_full_output(one_hot, (x, y), combined_one_hots[img_name],
                                                          orig_img_shape)

            # a third image was started -> we can save the first one
            else:
                # save the old one before starting the new one
                img_to_save = current_img_names.pop(0)
                one_hot_finished = combined_one_hots.pop(img_to_save)
                pred, target, mean_iu = _save_test_img_output(img_to_save, one_hot_finished, multi_run, dataset_folder, logging_label, writer, epoch, num_classes, use_boundary_pixel)
                preds.append(pred)
                targets.append(target)
                # update the meanIU
                meanIU.update(mean_iu, 1)

                # start the combination of the new image
                logging.info("Starting segmentation of image {}".format(img_name))
                combined_one_hots[img_name] = one_hot_to_full_output(one_hot, (x, y), combined_one_hots[img_name],
                                                          orig_img_shape)

    # save all the remaining images
    while len(current_img_names) > 0:
        img_to_save = current_img_names.pop(0)
        one_hot_finished = combined_one_hots.pop(img_to_save)
        pred, target, mean_iu = _save_test_img_output(img_to_save, one_hot_finished, multi_run, dataset_folder, logging_label, writer, epoch, num_classes, use_boundary_pixel)
        preds.append(pred)
        targets.append(target)
        # update the meanIU
        meanIU.update(mean_iu, 1)

    # Make a confusion matrix
    try:
        #targets_flat = np.array(targets).flatten()
        #preds_flat = np.array(preds).flatten()
        # load the weights
        weights = _load_class_frequencies_weights_from_file(dataset_folder, inmem, workers, runner_class)
        # calculate the confusion matrix
        cm = confusion_matrix(y_true=np.array(targets).flatten(), y_pred=np.array(preds).flatten(), labels=[i for i in range(num_classes)])
        cm_w = confusion_matrix(y_true=np.array(targets).flatten(), y_pred=np.array(preds).flatten(),
                                labels=[i for i in range(num_classes)], sample_weight=[weights[i] for i in np.array(targets).flatten()])
        confusion_matrix_heatmap = make_heatmap(cm, class_names)
        confusion_matrix_heatmap_w = make_heatmap(np.round(cm_w*100).astype(np.int), class_names)

    except ValueError:
        logging.warning('Confusion Matrix did not work as expected')
        confusion_matrix_heatmap = np.zeros((10, 10, 3))
        confusion_matrix_heatmap_w = confusion_matrix_heatmap

    # Logging the epoch-wise accuracy and saving the confusion matrix
    if multi_run is None:
        writer.add_scalar(logging_label + '/meanIU', meanIU.avg, epoch)
        save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix',
                                          image=confusion_matrix_heatmap, global_step=epoch)
        save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix_weighted',
                                          image=confusion_matrix_heatmap_w, global_step=epoch)
    else:
        writer.add_scalar(logging_label + '/meanIU_{}'.format(multi_run), meanIU.avg, epoch)
        save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix_{}'.format(multi_run),
                                          image=confusion_matrix_heatmap, global_step = epoch)
        save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix_weighted{}'.format(multi_run),
                                          image=confusion_matrix_heatmap_w, global_step=epoch)


    logging.info(_prettyprint_logging_label(logging_label) +
                 ' epoch[{}]: '
                 'MeanIU={meanIU.avg:.3f}\t'
                 'Loss={loss.avg:.4f}\t'
                 'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                 .format(epoch, batch_time=batch_time, data_time=data_time, loss=losses, meanIU=meanIU))

    # # Generate a classification report for each epoch
    # _log_classification_report(data_loader, epoch, preds, targets, writer)

    return meanIU.avg


def _save_test_img_output(img_to_save, one_hot, multi_run, dataset_folder, logging_label, writer, epoch, num_classes, use_boundary_pixel):
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

    Returns
    -------
    pred, target: numpy arrays
        argmax of the predicted and target values for the image
    """
    logging.info("Finished segmentation of image {}. Saving output...".format(img_to_save))
    np_bgr = one_hot_to_np_bgr(one_hot)
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

    # Compute and record the meanIU of the whole image TODO check with Vinay & Michele if correct
    acc, acc_cls, mean_iu, fwavacc = accuracy_segmentation(target, pred, num_classes)
    logging.info("MeanIU {}: {}".format(img_to_save, mean_iu))
    if use_boundary_pixel:
        pred[border_mask] = target[border_mask]
    acc, acc_cls, mean_iu, fwavacc = accuracy_segmentation(target, pred, num_classes)
    txt = " (adjusted for the boundary pixel)" if use_boundary_pixel else ""
    logging.info("MeanIU {}: {}{}".format(img_to_save, mean_iu, txt))

    # TODO: also save input and gt image?
    if multi_run is None:
        # writer.add_scalar(logging_label + '/meanIU', mean_iu, epoch)
        save_image_and_log_to_tensorboard_segmentation(writer, tag=logging_label + '/output_{}'.format(img_to_save),
                                                       image=np_bgr,
                                                       gt_image=ground_truth[:, :, ::-1])  # ground_truth[:, :, ::-1] convert image to BGR
    else:
        # writer.add_scalar(logging_label + '/meanIU_{}'.format(multi_run), mean_iu, epoch)
        save_image_and_log_to_tensorboard_segmentation(writer, tag=logging_label + '/output_{}_{}'.format(multi_run,
                                                                                                          img_to_save),
                                                       image=np_bgr,
                                                       gt_image=ground_truth[:, :, ::-1])  # ground_truth[:, :, ::-1] convert image to BGR

    return pred, target, mean_iu


def _log_classification_report(data_loader, epoch, preds, targets, writer):
    """
    This routine computes and prints on Tensorboard TEXT a classification
    report with F1 score, Precision, Recall and similar metrics computed
    per-class.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set
    epoch : int
        Number of the epoch (for logging purposes)
    preds : list
        List of all predictions of the model for this epoch
    targets : list
        List of all correct labels for this epoch
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.

    Returns
    -------
        None
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        classification_report_string = str(classification_report(y_true=targets,
                                                                 y_pred=preds,
                                                                 target_names=[str(item) for item in
                                                                               data_loader.dataset.classes]))
    # Fix for TB writer. Its an ugly workaround to have it printed nicely in the TEXT section of TB.
    classification_report_string = classification_report_string.replace('\n ', '\n\n       ')
    classification_report_string = classification_report_string.replace('precision', '      precision', 1)
    classification_report_string = classification_report_string.replace('avg', '      avg', 1)

    writer.add_text('Classification Report for epoch {}\n'.format(epoch), '\n' + classification_report_string, epoch)

def save_image_and_log_to_tensorboard_segmentation(writer=None, tag=None, image=None, global_step=None, gt_image=[]):
    """Utility function to save image in the output folder and also log it to Tensorboard.
    ALL IMAGES ARE IN BGR BECAUSE OF CV3.IMWRITE()!!

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
        dest_filename = os.path.join(output_folder, 'images', tag + '_{}'.format(global_step))
    else:
        dest_filename = os.path.join(output_folder, 'images', tag)

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
        dest_filename = os.path.join(output_folder, 'images', tag_col + '_{}'.format(global_step))
    else:
        dest_filename = os.path.join(output_folder, 'images', tag_col)

    if not os.path.exists(os.path.dirname(dest_filename)):
        os.makedirs(os.path.dirname(dest_filename))

    img = np.copy(image)
    blue = image[:, :, 0]  # Extract just blue channel
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
    if len(gt_image) != 0:
        class_col = {"fg_correct": (30, 160, 70), "fg_wrong_class": (60, 255, 255), "bg_correct": (0, 0, 0),
                     "bg_as_fg": (20, 30, 240), "fg_as_bg": (255, 240, 0)}

        img_la = np.copy(image)
        tag_la = "layout_analysis_" + tag
        # Get output folder using the FileHandler from the logger.
        # (Assumes the file handler is the last one)
        output_folder = os.path.dirname(logging.getLogger().handlers[-1].baseFilename)

        if global_step is not None:
            dest_filename = os.path.join(output_folder, 'images', tag_la + '_{}'.format(global_step))
        else:
            dest_filename = os.path.join(output_folder, 'images', tag_la)

        if not os.path.exists(os.path.dirname(dest_filename)):
            os.makedirs(os.path.dirname(dest_filename))

        out_blue = image[:, :, 0]  # Extract just blue channel
        gt_blue = gt_image[:, :, 0]

        masks = {c: _get_mask(c, out_blue, gt_blue) for c in class_col.keys()}

        for c, mask in masks.items():
            img_la[mask] = class_col[c]

        # img_la = np.array([[_get_colour(out_blue[x, y], gt_blue[x, y]) for y in range(out_blue.shape[1])]
        #                    for x in range(out_blue.shape[0])])
        # Write image to output folder
        # TODO this does not work
        #writer.add_image(tag=tag_la, img_tensor=np.moveaxis(img_la, -1, 0), global_step=global_step)
        cv2.imwrite(dest_filename, img_la)

    return

def _get_mask(tag, output, gt):
    if tag == "fg_correct":
        return np.logical_and(output == gt, gt != 1)
    elif tag == "bg_correct":
        return np.logical_and(output == gt, gt == 1)
    elif tag == "fg_wrong_class":
        return np.logical_and(output != gt, np.logical_and(gt != 1, output != 1))
    elif tag == "fg_as_bg":
        return np.logical_and(output != gt, output == 1)
    elif tag == "bg_as_fg":
        return np.logical_and(output != gt, np.logical_and(output != 1, gt == 1))


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