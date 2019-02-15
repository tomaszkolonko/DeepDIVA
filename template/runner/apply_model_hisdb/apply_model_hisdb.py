"""
This file is the template for the boilerplate of train/test of a DNN for image classification

There are a lot of parameter which can be specified to modify the behaviour and they should be used 
instead of hard-coding stuff.
"""

import logging
import sys
import os

# Utils
import numpy as np

# DeepDIVA
import torch
from torch import nn

import models
# Delegated
from template.runner.apply_model_hisdb import evaluate
from template.setup import set_up_model
from .setup import set_up_dataloader
from util.misc import checkpoint, adjust_learning_rate


class ApplyModelHisdb:
    @staticmethod
    def single_run(writer, current_log_folder, model_name, epochs, lr, decay_lr,
                   validation_interval, checkpoint_all_epochs,
                   input_patch_size, **kwargs):
        """
        This is the main routine where train(), validate() and test() are called.

        Parameters
        ----------
        writer : Tensorboard.SummaryWriter
            Responsible for writing logs in Tensorboard compatible format.
        current_log_folder : string
            Path to where logs/checkpoints are saved
        model_name : string
            Name of the model
        epochs : int
            Number of epochs to train
        lr : float
            Value for learning rate
        kwargs : dict
            Any additional arguments.
        decay_lr : boolean
            Decay the lr flag
        validation_interval : int
            Run evaluation on validation set every N epochs
        checkpoint_all_epochs : bool
            If enabled, save checkpoint after every epoch.
        input_patch_size : int
            Size of the input patch, e.g. with 32 the input will be re-sized to 32x32

        Returns
        -------
        train_value : ndarray[floats] of size (1, `epochs`)
            Accuracy values for train split
        val_value : ndarray[floats] of size (1, `epochs`+1)
            Accuracy values for validation split
        test_value : float
            Accuracy value for test split
        """

        int_val_to_class_name = {1: "background", 2: "comment", 4: "decoration", 6: "comment_decoration",
                                 8: "maintext", 10: "maintext_comment", 12: "maintext_decoration",
                                 14: "maintext_comment_decoration"}

        class_names = [v for k, v in int_val_to_class_name.items()]
        num_classes = len(class_names)

        # Setting up the dataloaders
        aply_ds_loader = set_up_dataloader(input_patch_size, **dict(kwargs, num_classes=num_classes))

        logging.info('Loading the specified model and evaluating on the provided data set')

        # TODO: add weights to kwargs
        model, criterion, _, _, _ = set_up_model(num_classes=num_classes,
                                         model_name=model_name,
                                         lr=lr,
                                         train_loader=None,
                                         **kwargs)

        # Test
        meanIU = ApplyModelHisdb._apply_model(aply_ds_loader, model, criterion, writer, epochs - 1, class_names, **kwargs)
        logging.info('Segmentation completed')

        return meanIU

    ####################################################################################################################
    @staticmethod
    def _validate_model_input_size(model_expected_input_size, model_name):
        """
        This method verifies that the model expected input size is a tuple of 2 elements.
        This is necessary to avoid confusion with models which run on other types of data.

        Parameters
        ----------
        model_expected_input_size
            The item retrieved from the model which corresponds to the expected input size
        model_name : String
            Name of the model (logging purpose only)

        Returns
        -------
            None
        """
        if type(model_expected_input_size) is not tuple or len(model_expected_input_size) != 2:
            logging.error('Model {model_name} expected input size is not a tuple. '
                          'Received: {model_expected_input_size}'
                          .format(model_name=model_name,
                                  model_expected_input_size=model_expected_input_size))
            sys.exit(-1)

    ####################################################################################################################
    """
    These methods delegate their function to other classes in this package. 
    It is useful because sub-classes can selectively change the logic of certain parts only.
    """

    @classmethod
    def _apply_model(cls, test_loader, model, criterion, writer, epoch, class_names, **kwargs):
        return evaluate.apply(test_loader, model, criterion, writer, epoch, class_names, **kwargs)
