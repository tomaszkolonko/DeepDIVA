# Utils
import argparse
import inspect
import os
import shutil
import re
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

def hisDB(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the HisDB-all dataset for semantic segmentation to the location specified
    on the file system

    Output folder structure: ../HisDB/CB55/train
                             ../HisDB/CB55/val
                             ../HisDB/CB55/test

                             ../HisDB/CB55/test/data -> images
                             ../HisDB/CB55/test/gt   -> pixel-wise annotated ground truth

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
    link = urllib.parse.urlparse(
        'https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/hisdoc/all.zip')
    # link_test_private = urllib.parse.urlparse(
    #    'https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/hisdoc/private-test/all-privateTest.zip')
    download_path = os.path.join(dataset_root, link.geturl().rsplit('/', 1)[-1])

    # download files
    print('Downloading {}...'.format(link.geturl()))
    urllib.request.urlretrieve(link.geturl(), download_path)
    print('Download complete. Unpacking files...')

    # unpack relevant folders
    zip_file = zipfile.ZipFile(download_path)

    # unpack imgs and gt
    data_gt_zip = {f: re.sub(r'img', 'pixel-level-gt', f) for f in zip_file.namelist() if 'img' in f}
    dataset_folders = [data_file.split('-')[-1][:-4] for data_file in data_gt_zip.keys()]
    for data_file, gt_file in data_gt_zip.items():
        dataset_name = data_file.split('-')[-1][:-4]
        dataset_folder = os.path.join(dataset_root, dataset_name)
        _make_folder_if_not_exists(dataset_folder)

        for file in [data_file, gt_file]:
            zip_file.extract(file, dataset_folder)
            with zipfile.ZipFile(os.path.join(dataset_folder, file), "r") as zip_ref:
                zip_ref.extractall(dataset_folder)
                # delete zips
                os.remove(os.path.join(dataset_folder, file))

        # create folder structure
        for partition in ['train', 'val', 'test']:
            for folder in ['data', 'gt']:
                _make_folder_if_not_exists(os.path.join(dataset_folder, partition, folder))

    # move the files to the correct place
    for folder in dataset_folders:
        for k1, v1 in {'pixel-level-gt': 'gt', 'img': 'data'}.items():
            for k2, v2 in {'public-test': 'test', 'training': 'train', 'validation': 'val'}.items():
                current_path = os.path.join(dataset_root, folder, k1, k2)
                new_path = os.path.join(dataset_root, folder, v2, v1)
                for f in [f for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]:
                    shutil.move(os.path.join(current_path, f), os.path.join(new_path, f))
            # remove old folders
            shutil.rmtree(os.path.join(dataset_root, folder, k1))
    print('Finished. Data set up at {}.'.format(dataset_root))


def icdar2017_clamm(args):

    url = "http://clamm.irht.cnrs.fr/wp-content/uploads/ICDAR2017_CLaMM_Training.zip"
    print("Downloading " + url)
    zip_name = "ICDAR2017_CLaMM_Training.zip"
    local_filename, headers = urllib.request.urlretrieve(url, zip_name)
    zfile = zipfile.ZipFile(local_filename)

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'ICDAR2017-CLAMM')
    dataset_manuscriptDating = os.path.join(dataset_root , 'ManuscrpitDating')
    dataset_md_train = os.path.join(dataset_manuscriptDating , 'train')
    dataset_styleClassification = os.path.join(dataset_root , 'StyleClassification')
    dataset_sc_train = os.path.join(dataset_styleClassification, 'train')
    test_sc_folder = os.path.join(dataset_styleClassification, 'test')
    test_md_folder = os.path.join(dataset_manuscriptDating, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(dataset_manuscriptDating)
    _make_folder_if_not_exists(dataset_styleClassification)
    _make_folder_if_not_exists(test_sc_folder)

    def _write_data_to_folder(zipfile, filenames, labels, folder, start_index,  isTest):

        sorted_labels = [None]*len(labels)
        if isTest == 1:
            for i in range(len(zipfile.infolist())):
                entry = zipfile.infolist()[i]
                if "IRHT_P_009793.tif" in entry.filename:
                    zipfile.infolist().remove(entry)
                    break

        zip_infolist = zipfile.infolist()[1:]

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
                img.save(os.path.join(dest, str(i) + '.png'), "PNG", quality=100)

    def getLabels(zfile):
        filenames, md_labels, sc_labels = [], [], []
        zip_infolist = zfile.infolist()[1:]
        for entry in zip_infolist:
            if '.csv' in entry.filename:
                with zfile.open(entry) as file:
                    cf = file.read()
                    c = csv.StringIO(cf.decode())
                    next(c) # Skip the first line which is the header of csv file
                    for row in c:

                        md_label_strt_ind = row.rfind(';')
                        md_label_end_ind = row.rfind("\r")
                        md_labels.append(row[md_label_strt_ind+1:md_label_end_ind])
                        sc_labels_strt_ind = row[:md_label_strt_ind].rfind(';')
                        sc_labels.append(row[sc_labels_strt_ind+1:md_label_strt_ind])
                        filename_ind = row[:sc_labels_strt_ind].rfind(';')

                        if filename_ind > -1:
                            f_name = row[filename_ind+1:sc_labels_strt_ind]
                        else:
                            f_name = row[:sc_labels_strt_ind]
                        if isTest == 1 and f_name == 'IRHT_P_009783.tif':
                            print('No file named ' + f_name + ". This filename will not be added!")
                        else:
                            filenames.append(f_name)

                zfile.infolist().remove(entry) # remove the csv file from infolist
            if '.db' in entry.filename: # remove the db file from infolist
                zfile.infolist().remove(entry)
        return filenames, sc_labels, md_labels

    isTest = 0
    filenames, sc_labels, md_labels = getLabels(zfile)
    start_index_training = len("ICDAR2017_CLaMM_Training/")

    _write_data_to_folder(zfile, filenames, sc_labels, dataset_sc_train, start_index_training, isTest)
    _write_data_to_folder(zfile, filenames, md_labels, dataset_md_train, start_index_training, isTest)

    os.remove(os.path.join(zfile.filename))

    url = "http://clamm.irht.cnrs.fr/wp-content/uploads/ICDAR2017_CLaMM_task1_task3.zip"
    print("Downloading " + url)
    zip_name_test = "ICDAR2017_CLaMM_task1_task3.zip"
    local_filename_test, headers_test = urllib.request.urlretrieve(url, zip_name_test)
    zfile_test = zipfile.ZipFile(local_filename_test)

    isTest = 1
    filenames_test, sc_test_labels, md_test_labels = getLabels(zfile_test)
    start_index_test = len("ICDAR2017_CLaMM_task1_task3/")
    _write_data_to_folder(zfile_test, filenames_test, sc_test_labels, test_sc_folder, start_index_test, 1)
    _write_data_to_folder(zfile_test, filenames_test, md_test_labels, test_md_folder, start_index_test, 1)

    os.remove(os.path.join(zfile_test.filename))

    split_dataset(dataset_folder=dataset_manuscriptDating, split=0.2, symbolic=False)
    split_dataset(dataset_folder=dataset_styleClassification, split=0.2, symbolic=False)
    print("ICDAR2017 CLaMM data is ready!")


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
