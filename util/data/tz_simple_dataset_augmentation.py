"""
This script allows to crop the asbestos images into 256x256 images for input into alexnet
"""

# Utils
import argparse
import os
from PIL import Image


def get_all_file_names(dataset_folder):
    """
    Get all file names recursively below the provided dataset folder

    :param dataset_folder: path to the dataset folder
    :return: list of file names
    """
    recursive_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(dataset_folder) for f in fn if f.endswith('png')]
    return recursive_files


def do_augmentation(file_names):
    """
    initialize the augmentation. Call each function seperatly with the current file_name

    :param file_names: list of all file names below the provided dataset folder

    :return:
    """
    for file_name in file_names:
        try:
            flip_horizontally(file_name)
            flip_vertically(file_name)
            rotate_image(file_name)

            print("SUCCESS")
        except(IOError, KeyError) as e:
            print(e)


def rotate_image(file_name):
    """
    Rotate image and save it with new file name into same directory as original file

    :param file_name:
    :return:
    """
    for angle in [90, 180, 270]:
        image_obj = Image.open(file_name)
        rotated_image = image_obj.rotate(angle)
        new_file_name = file_name[:-4] + "_rotated_by_" + str(angle) + ".png"
        rotated_image.save(new_file_name)


def flip_vertically(file_name):
    """
    Flip image vertically and save it with new file name into same directory as original file

    :param file_name:
    :return:
    """
    image_obj = Image.open(file_name)
    flipped_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    new_file_name = file_name[:-4] + "_flipped_vertically.png"
    flipped_image.save(new_file_name)


def flip_horizontally(file_name):
    """
    Flip image horizontally and save it with new file name into same directory as original file

    :param file_name:
    :return:
    """
    image_obj = Image.open(file_name)
    flipped_image = image_obj.transpose(Image.FLIP_TOP_BOTTOM)
    new_file_name = file_name[:-4] + "_flipped_horizontally.png"
    flipped_image.save(new_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script rotates all images and saves them all')

    parser.add_argument('--dataset-folder',
                        help='path to the dataset.',
                        required=True,
                        type=str,
                        default=None)

    args = parser.parse_args()

    list_of_all_file_names = get_all_file_names(dataset_folder=args.dataset_folder)
    do_augmentation(list_of_all_file_names)
