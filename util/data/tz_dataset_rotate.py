"""
This script allows to crop the asbestos images into 256x256 images for input into alexnet
"""

# Utils
import argparse
import os
import numpy as np
from PIL import Image


from os import listdir
from os.path import isfile, join



def rotate_images(dataset_folder):
    """
    Resize all images contained within dataset_folder.

    Parameters
    ----------
    dataset_folder : str
        Path to the dataset folder

    Returns
    -------
        None
    """

    # Get all the files recursively from the given dataset_folder
    recursiveFiles = [os.path.join(dp, f) for dp, dn, fn in os.walk(dataset_folder) for f in fn if f.endswith('png')]

    print(recursiveFiles)

    size = 256, 256

    for infile in recursiveFiles:
        file_name = infile[:-4]
        try:
            for angle in [90, 180, 270]:
                image_obj = Image.open(infile)
                rotated_image = image_obj.rotate(angle)
                new_file_name = file_name + "_" + str(angle) + ".png"
                rotated_image.save(new_file_name)

            print("SUCCESS")
        except:
            print("FAIL")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script rotates all images and saves them all')

    parser.add_argument('--dataset-folder',
                        help='path to the dataset.',
                        required=True,
                        type=str,
                        default=None)

    args = parser.parse_args()

    rotate_images(dataset_folder=args.dataset_folder)
