"""
This script parsed all images and prints the average width and heigth
"""

# Utils
import argparse
import os
from PIL import Image

from os import listdir
from os.path import isfile, join



def print_average_width_length(dataset_folder):
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
    total_width = 0
    total_length = 0
    counter = 0

    # Get all the files recursively from the given dataset_folder
    recursiveFiles = [os.path.join(dp, f) for dp, dn, fn in os.walk(dataset_folder) for f in fn if f.endswith('png')]

    for infile in recursiveFiles:
        try:
            counter += 1
            im = Image.open(infile)
            width, length = im.size
            total_width += width
            total_length += length
        except:
            print("FAIL");
    print("average width: " + str(total_width / counter))
    print("average length: " + str(total_length / counter))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script averages the widths and lengths over all images')

    parser.add_argument('--dataset-folder',
                        help='path to the dataset.',
                        required=True,
                        type=str,
                        default=None)

    args = parser.parse_args()

    print_average_width_length(dataset_folder=args.dataset_folder)
