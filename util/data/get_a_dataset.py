# Utils
import argparse
import fnmatch
import inspect
import os
import shutil
import sys

import numpy as np
import scipy


import zipfile
import urllib.request
import csv

# Torch
import torch
import torchvision
from PIL import Image

# DeepDIVA
from util.data.dataset_splitter import split_dataset


def mnist(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the MNIST dataset to the location specified
    on the file system

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    # Use torchvision to download the dataset
    torchvision.datasets.MNIST(root=args.output_folder, download=True)

    # Load the data into memory
    train_data, train_labels = torch.load(os.path.join(args.output_folder,
                                                       'processed',
                                                       'training.pt'))
    test_data, test_labels = torch.load(os.path.join(args.output_folder,
                                                     'processed',
                                                     'test.pt'))

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'MNIST')
    train_folder = os.path.join(dataset_root, 'train')
    test_folder = os.path.join(dataset_root, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(train_folder)
    _make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = os.path.join(folder, str(label))
            _make_folder_if_not_exists(dest)
            Image.fromarray(img.numpy(), mode='L').save(os.path.join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    shutil.rmtree(os.path.join(args.output_folder, 'raw'))
    shutil.rmtree(os.path.join(args.output_folder, 'processed'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)


def svhn(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the SVHN dataset to the location specified
    on the file system

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    # Use torchvision to download the dataset
    torchvision.datasets.SVHN(root=args.output_folder, split='train', download=True)
    torchvision.datasets.SVHN(root=args.output_folder, split='test', download=True)

    # Load the data into memory
    train = scipy.io.loadmat(os.path.join(args.output_folder,
                                          'train_32x32.mat'))
    train_data, train_labels = train['X'], train['y'].astype(np.int64).squeeze()
    np.place(train_labels, train_labels == 10, 0)
    train_data = np.transpose(train_data, (3, 0, 1, 2))

    test = scipy.io.loadmat(os.path.join(args.output_folder,
                                         'test_32x32.mat'))
    test_data, test_labels = test['X'], test['y'].astype(np.int64).squeeze()
    np.place(test_labels, test_labels == 10, 0)
    test_data = np.transpose(test_data, (3, 0, 1, 2))

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'SVHN')
    train_folder = os.path.join(dataset_root, 'train')
    test_folder = os.path.join(dataset_root, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(train_folder)
    _make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = os.path.join(folder, str(label))
            _make_folder_if_not_exists(dest)
            Image.fromarray(img).save(os.path.join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    os.remove(os.path.join(args.output_folder, 'train_32x32.mat'))
    os.remove(os.path.join(args.output_folder, 'test_32x32.mat'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)


def cifar10(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the CIFAR dataset to the location specified
    on the file system

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    # Use torchvision to download the dataset
    cifar_train = torchvision.datasets.CIFAR10(root=args.output_folder, train=True, download=True)
    cifar_test = torchvision.datasets.CIFAR10(root=args.output_folder, train=False, download=True)

    # Load the data into memory
    train_data, train_labels = cifar_train.train_data, cifar_train.train_labels

    test_data, test_labels = cifar_test.test_data, cifar_test.test_labels

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'CIFAR10')
    train_folder = os.path.join(dataset_root, 'train')
    test_folder = os.path.join(dataset_root, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(train_folder)
    _make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = os.path.join(folder, str(label))
            _make_folder_if_not_exists(dest)
            Image.fromarray(img).save(os.path.join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    os.remove(os.path.join(args.output_folder, 'cifar-10-python.tar.gz'))
    shutil.rmtree(os.path.join(args.output_folder, 'cifar-10-batches-py'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)


def icdar2017_clamm(args):

    url = "http://clamm.irht.cnrs.fr/wp-content/uploads/ICDAR2017_CLaMM_Training.zip"
    print("Downloading " + url)
    zip_name = "ICDAR2017_CLaMM_Training.zip"
    local_filename, headers = urllib.request.urlretrieve(url, zip_name)
    zfile = zipfile.ZipFile(local_filename)

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'ICDAR2017-CLAMM')
    dataset_manuscriptDating = os.path.join(dataset_root, 'ManuscrpitDating')
    dataset_styleClassification = os.path.join(dataset_root, 'StyleClassification')
    #test_folder = os.path.join(dataset_root, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(dataset_manuscriptDating)
    _make_folder_if_not_exists(dataset_styleClassification)
    #_make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(zipfile, filenames, labels, folder):

        sorted_labels = [None]*len(labels)
        zip_infolist = zipfile.infolist()[1:]
        start_index = len("ICDAR2017_CLaMM_Training/")
        for i in range(len(zip_infolist)):
            entry = zip_infolist[i]
            entry_index_infilenames = filenames.index(entry.filename[start_index:])
            sorted_labels[i] = labels[entry_index_infilenames]

        print(zip_infolist[1].filename)
        print(sorted_labels[1])

        for i, (enrty, label) in enumerate(zip(zipfile.infolist()[1:], sorted_labels)):
            with zipfile.open(enrty) as file:
                img = Image.open(file)
                dest = os.path.join(folder, str(label))
                _make_folder_if_not_exists(dest)
                img.save(os.path.join(dest, str(i) + '.tif'))

    filenames, md_labels, sc_labels = [], [], []
    for entry in zfile.infolist():
        if '.csv' in entry.filename:
            with zfile.open(entry) as file:
                cf = file.read()
                c = csv.StringIO(cf.decode())
                next(c) # Skip the first line which is the header of csv file
                for row in c:
                    filename_ind = row.find(';')
                    filenames.append(row[0:filename_ind])
                    sc_label_ind = row[filename_ind+1:].find(';')
                    sc_labels.append(int(row[filename_ind+1:filename_ind+1+sc_label_ind]))
                    md_label_end_ind = row[filename_ind+1+sc_label_ind+1:].find("\r")
                    md_labels.append(int(row[filename_ind+1+sc_label_ind+1:filename_ind+1+sc_label_ind+1+md_label_end_ind]))

            zfile.infolist().remove(entry) # remove the csv file from infolist
        if '.db' in entry.filename:
            zfile.infolist().remove(entry)

    _write_data_to_folder(zfile, filenames, sc_labels, dataset_styleClassification)
    _write_data_to_folder(zfile, filenames, md_labels, dataset_manuscriptDating)
    os.remove(os.path.join(zfile.filename))
    print("ICDAR2017 CLaMM data is ready!")


def historical_wi(args):

    binarizedData_url = "ftp://scruffy.caa.tuwien.ac.at/staff/database/icdar2017/icdar17-historicalwi-training-binarized.zip"
    gt_url = "ftp://scruffy.caa.tuwien.ac.at/staff/database/icdar2017/icdar17-historicalwi-training-color.zip"
    zip_name = "icdar17-historicalwi-training-binarized.zip"
    zip_name_gt = "icdar17-historicalwi-training-color.zip"

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'Historical_WI')
    train_folder = os.path.join(dataset_root, 'train')
    gt_folder = os.path.join(dataset_root, 'GT')
    # test_folder = os.path.join(dataset_root, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(train_folder)
    _make_folder_if_not_exists(gt_folder)
    # _make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(zipfile, labels, folder):

        for i, (enrty, label) in enumerate(zip(zipfile.infolist()[1:], labels)):
            with zipfile.open(enrty) as file:
                img = Image.open(file)
                dest = os.path.join(folder, str(label))
                _make_folder_if_not_exists(dest)
                img.save(os.path.join(dest, str(i) + '.png'))

    def _get_labels(zipfile, start_index):
        labels = []
        for zipinfo in zipfile.infolist()[1:]:
            file_name = zipinfo.filename
            ind = file_name.find("-", start_index)
            labels.append(file_name[start_index:ind])
        return labels

    # For binarized dataset
    print("Downloading " + binarizedData_url)
    local_filename, headers = urllib.request.urlretrieve(binarizedData_url, zip_name)
    zfile = zipfile.ZipFile(local_filename)
    training_labels = _get_labels(zfile, len("icdar2017-training-binary/"))
    _write_data_to_folder(zfile, training_labels, train_folder)
    os.remove(os.path.join(zfile.filename))
    print("Binary data is ready!")

    # For colored dataset
    print("Downloading " + gt_url)
    local_filename_gt, headers_gt = urllib.request.urlretrieve(gt_url, zip_name_gt)
    zfile_gt = zipfile.ZipFile(local_filename_gt)
    gt_labels = _get_labels(zfile_gt, len("icdar2017-training-color/"))
    _write_data_to_folder(zfile_gt, gt_labels, gt_folder)
    os.remove(os.path.join(zfile_gt.filename))
    print("Colored data is ready!")

def _make_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    downloadable_datasets = [name[0] for name in inspect.getmembers(sys.modules[__name__],
                                                                    inspect.isfunction) if not name[0].startswith('_')]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script can be used to download some '
                                                 'datasets and prepare them in a standard format')

    parser.add_argument('--dataset',
                        help='name of the dataset',
                        type=str,
                        choices=downloadable_datasets)
    parser.add_argument('--output-folder',
                        help='path to where the dataset should be generated.',
                        required=False,
                        type=str,
                        default='./data/')
    args = parser.parse_args()

    getattr(sys.modules[__name__], args.dataset)(args)
