"""
This script allows to crop the asbestos images into 256x256 images for input into alexnet
"""

# Utils
import argparse
import os
from PIL import Image

from os import listdir
from os.path import isfile, join



def crop_images(dataset_folder):
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
    recursiveFiles = [os.path.join(dp, f) for dp, dn, fn in os.walk(dataset_folder) for f in fn if f.endswith('jpg')]

    print(recursiveFiles)

    size = 1200, 1800

    for infile in recursiveFiles:
        try:
            im = Image.open(infile)
            out = im.crop((100, 100, 1300, 1900))
            print("")
            out.save(infile, "png")


            print("SUCCESS")
        except:
            print("FAIL")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script crops all images and adds them to the folder')

    parser.add_argument('--dataset-folder',
                        help='path to the dataset.',
                        required=True,
                        type=str,
                        default=None)

    args = parser.parse_args()

    crop_images(dataset_folder=args.dataset_folder)
