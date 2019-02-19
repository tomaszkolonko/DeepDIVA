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
from datasets.transform_library.functional import annotation_to_argmax
from util.visualization.confusion_matrix_heatmap import make_heatmap
from util.evaluation.metrics.accuracy import accuracy_segmentation


def evaluate(logging_label, data_loader, model, criterion, writer, epoch, name_onehotindex, category_id_name,
             no_val_conf_matrix, no_cuda=False, log_interval=10, **kwargs):
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

    num_classes = len(name_onehotindex)
    class_names = sorted(name_onehotindex, key=name_onehotindex.get)
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
        # # convert input to torch tensor
        # input = torch.LongTensor(np.array([np.array(i) for i in input_batch]))
        #
        # # convert annotation to argmax
        # target_argmax = torch.LongTensor(np.array([annotation_to_argmax((img_pil.height, img_pil.width), annotations, name_onehotindex, category_id_name)
        #                                   for img_pil, annotations in zip(input_batch, annotations)]))

        # Measure data loading time
        data_time.update(time.time() - end)

        # Moving data to GPU
        if not no_cuda:
            input = input.cuda(async=True)
            target_argmax = target.cuda(async=True)

        # Convert the input and its labels to Torch Variables
        input_var = torch.autograd.Variable(input)
        target_argmax_var = torch.autograd.Variable(target_argmax)

        # Compute output
        output = model(input_var)
        output_argmax = np.array([np.argmax(o, axis=0) for o in output.data.cpu().numpy()])

        # Compute and record the loss
        loss = criterion(output, target_argmax_var)
        #losses.update(loss.data[0], input.size(0))
        losses.update(loss.item(), input.size(0))

        # Compute and record the accuracy TODO check with Vinay & Michele if correct
        acc, acc_cls, mean_iu, fwavacc = accuracy_segmentation(target_argmax.cpu().numpy(), output_argmax, num_classes)
        meanIU.update(mean_iu, input.size(0))

        # Get the predictions
        _ = [preds.append(item) for item in output_argmax]
        _ = [targets.append(item) for item in target_argmax.cpu().numpy()]

        # Add loss and accuracy to Tensorboard
        if multi_run is None:
            writer.add_scalar(logging_label + '/mb_loss', loss.item(), epoch * len(data_loader) + batch_idx)
            writer.add_scalar(logging_label + '/mb_meanIU', mean_iu, epoch * len(data_loader) + batch_idx)
        else:
            writer.add_scalar(logging_label + '/mb_loss_{}'.format(multi_run), loss.item(),
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

    # Make a confusion matrix
    if not no_val_conf_matrix:
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
    else:
        logging.info("No confusion matrix created.")

    # Logging the epoch-wise accuracy and saving the confusion matrix
    if multi_run is None:
        writer.add_scalar(logging_label + '/meanIU', meanIU.avg, epoch)
        if not no_val_conf_matrix :
            save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix',
                                              image=confusion_matrix_heatmap, global_step=epoch)
            # save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix_weighted',
            #                                   image=confusion_matrix_heatmap_w, global_step=epoch)
    else:
        writer.add_scalar(logging_label + '/meanIU_{}'.format(multi_run), meanIU.avg, epoch)
        if not no_val_conf_matrix :
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
