# Utils
import logging
import time
import warnings
import numpy as np

# Torch related stuff
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# DeepDIVA
from util.misc import AverageMeter, _prettyprint_logging_label, save_image_and_log_to_tensorboard, tensor_to_image,\
    one_hot_to_np_rgb, one_hot_to_full_output
from util.visualization.confusion_matrix_heatmap import make_heatmap
from util.evaluation.metrics.accuracy import accuracy_segmentation


def validate(val_loader, model, criterion, writer, epoch, class_names, no_cuda=False, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to validate the model."""
    return _evaluate(val_loader, model, criterion, writer, epoch, class_names, 'val', no_cuda, log_interval, **kwargs)


def test(test_loader, model, criterion, writer, epoch, class_names, no_cuda=False, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to test the model"""
    return _evaluate(test_loader, model, criterion, writer, epoch, class_names, 'test', no_cuda, log_interval, **kwargs)


def _evaluate(data_loader, model, criterion, writer, epoch, class_names, logging_label, no_cuda=False, log_interval=10, **kwargs):
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
    logging_label : string
        Label for logging purposes. Typically 'test' or 'valid'. Its prepended to the logging output path and messages.
    no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.
    log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    Returns
    -------
    meanIU.avg : float
        Accuracy of the model of the evaluated split
    """
    #TODO All parts computing the accuracy are commented out. It is necessary to
    #TODO implement a 2D softmax and instead of regressing the output class have it
    #TODO work with class labels. Notice that, however, it would be
    #TODO of interest leaving open the possibility to work with soft labels
    #TODO (e.g. the ground truth for pixel X,Y is an array of probabilities instead
    #TODO of an integer.
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

    combined_one_hot = []  # only needed for test phase
    current_output_filepath = ""  # only needed for test phase
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), unit='batch', ncols=150, leave=False)
    for batch_idx, (input, target) in pbar:
        # get_item returns more during "test", as the output of a whole image needs to be combined
        if logging_label == "test":
            input, orig_img_shape, top_left_coordinates, output_filepath = input

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

        # Compute and record the accuracy TODO
        acc, acc_cls, mean_iu, fwavacc = accuracy_segmentation(target_argmax.cpu().numpy(), output_argmax, num_classes)
        meanIU.update(mean_iu, input.size(0))

        # Get the predictions TODO
        _ = [preds.append(item) for item in output_argmax]
        _ = [targets.append(item) for item in target_argmax.cpu().numpy()]

        # Add loss and accuracy to Tensorboard
        if multi_run is None:
            writer.add_scalar(logging_label + '/mb_loss', loss.data[0], epoch * len(data_loader) + batch_idx)
            writer.add_scalar(logging_label + '/mb_accuracy', mean_iu, epoch * len(data_loader) + batch_idx)
        else:
            writer.add_scalar(logging_label + '/mb_loss_{}'.format(multi_run), loss.data[0],
                              epoch * len(data_loader) + batch_idx)
            writer.add_scalar(logging_label + '/mb_accuracy_{}'.format(multi_run), mean_iu,
                               epoch * len(data_loader) + batch_idx)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % log_interval == 0:
            pbar.set_description(logging_label +
                                 ' epoch [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(data_loader)))

            pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                             Loss='{loss.avg:.4f}\t'.format(loss=losses),
                             # Acc1='{meanIU.avg:.3f}\t'.format(meanIU=meanIU),
                             Data='{data_time.avg:.3f}\t'.format(data_time=data_time))

    # Make a confusion matrix TODO: ajust for class frequency
    try:
        targets_flat = np.array(targets).flatten()
        preds_flat = np.array(preds).flatten()
        # targets_flat = np.array([[np.argmax(a, axis=0) for a in target.numpy()] for target in targets]).flatten()
        # preds_flat = np.array([[np.argmax(a, axis=0) for a in pred.numpy()] for pred in preds]).flatten()
        cm = confusion_matrix(y_true=targets_flat, y_pred=preds_flat, labels=[i for i in range(num_classes)])
        confusion_matrix_heatmap = make_heatmap(cm, class_names) #TODO: continue here!!!
    except ValueError:
        logging.warning('Confusion Matrix did not work as expected')

        confusion_matrix_heatmap = np.zeros((10, 10, 3))

    # Logging the epoch-wise accuracy and saving the confusion matrix TODO: check accuracy measure
    if multi_run is None:
        writer.add_scalar(logging_label + '/accuracy', meanIU.avg, epoch)
        save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix',
                                          image=confusion_matrix_heatmap, global_step=epoch)
    else:
        writer.add_scalar(logging_label + '/accuracy_{}'.format(multi_run), meanIU.avg, epoch)
        save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix_{}'.format(multi_run))

    # Logging and saving the output images
    # if we are in testing, the output needs to be patched together to form the complete output of the full image
    # patches are returned as a sliding window over the full image, overlapping sections are averaged
    if logging_label == "test":
        # TODO change this so that there is a flag, when a new image is started
        if len(current_output_filepath) == 0 or output_filepath == current_output_filepath:
            # on the same image / first iteration
            combined_one_hot = one_hot_to_full_output(output.numpy(), top_left_coordinates, combined_one_hot,
                                                      orig_img_shape)
        else:
            # finished image, moving to next image
            # save the old one before starting the new one
            output = one_hot_to_np_rgb(combined_one_hot)
            # TODO: also save input and gt image

            if multi_run is None:
                writer.add_scalar(logging_label + '/accuracy', meanIU.avg, epoch)
                save_image_and_log_to_tensorboard(writer, tag=logging_label + '/output',
                                                  image=output, global_step=epoch)
            else:
                writer.add_scalar(logging_label + '/accuracy_{}'.format(multi_run), meanIU.avg, epoch)
                save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix_{}'.format(multi_run),
                                                  image=confusion_matrix_heatmap, global_step=epoch)
            # start the combination of the new image
            combined_one_hot = one_hot_to_full_output(output.numpy(), top_left_coordinates, [], orig_img_shape)

    logging.info(_prettyprint_logging_label(logging_label) +
                 ' epoch[{}]: '
                 'MeanIU={meanIU.avg:.3f}\t'
                 'Loss={loss.avg:.4f}\t'
                 'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                 .format(epoch, batch_time=batch_time, data_time=data_time, loss=losses, meanIU=meanIU))
    #
    # # Generate a classification report for each epoch TODO
    #_log_classification_report(data_loader, epoch, preds, targets, writer)

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
