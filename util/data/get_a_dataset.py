# Utils
import argparse
import inspect
import os
import shutil
import sys
import urllib
import zipfile

import numpy as np
import scipy
# Torch
import torch
import torchvision
from PIL import Image

# DeepDIVA
from util.data.dataset_splitter import split_dataset, split_dataset_segmentation


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

def hisDB(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the HisDB dataset for semantic segmentation to the location specified
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
    # make the root folder
    dataset_root = os.path.join(args.output_folder, 'HisDB')
    _make_folder_if_not_exists(dataset_root)

    # links to HisDB data sets
    linktrain_all = urllib.parse.urlparse(
        'https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/hisdoc/all.zip')
    linktest_all = urllib.parse.urlparse(
        'https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/hisdoc/private-test/all-privateTest.zip')

    d = {'train': linktrain_all, 'test': linktest_all}

    for data_use, link in d.items():
        download_path = os.path.join(dataset_root, link.geturl().rsplit('/', 1)[-1])

        # download files
        urllib.request.urlretrieve(link.geturl(), download_path)
        folder_extracted = os.path.splitext(os.path.join(dataset_root, link.geturl().rsplit('/', 1)[-1].rsplit('.', 1)[0]))[0]

        # unpack relevant folders
        zip_file = zipfile.ZipFile(download_path)

        # unpack imgs and gt
        for file in zip_file.namelist():
            if file.startswith('img'):
                _unpack_folder_hisDB(zip_file, file, folder_extracted, 'imgs', data_use)
            elif file.startswith('pixel'):
                _unpack_folder_hisDB(zip_file, file, folder_extracted, 'gt', data_use)

        os.rename(folder_extracted, os.path.join(os.path.dirname(folder_extracted), data_use))

        # split the training data set
        split_dataset_segmentation(dataset_folder=dataset_root, split=0.2, symbolic=False)


def _unpack_folder_hisDB(zip_file, file, folder_extracted, subfolder, data_use):
    """
    This function unpacks the images and the pixel-label gt of the HisDB from the zip files
    :param zip_file: mother-zip file to unpack
    :param file: zip-file in zip file to unpack
    :param folder_extracted: location, where the child-zips are extracted to
    :param subfolder: either 'imgs' or 'gt' (imgs for image input data, gt for the pixel-labelled images)
    :param data_use: either train or test
    :return: None
    """
    if data_use == 'test':
        dirname = os.path.join(folder_extracted, subfolder)
    elif data_use == 'train':
        dirname = os.path.join(folder_extracted, subfolder, file.rsplit('.', 1)[0].rsplit('-', 1)[-1])

    # unzip files
    zip_file.extract(file, dirname)
    with zipfile.ZipFile(os.path.join(dirname, file), "r") as zip_ref:
        zip_ref.extractall(dirname)
        # delete zips
        os.remove(os.path.join(dirname, file))

    if data_use == 'train':
        # move all images up
        for subdir, dirs, files in os.walk(dirname):
            for file in files:
                shutil.move(os.path.join(subdir, file), os.path.join(dirname, file))
        to_remove = {'gt': 'pixel-level-gt', 'imgs': 'img'}
        shutil.rmtree(os.path.join(dirname, to_remove[subfolder]))


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
