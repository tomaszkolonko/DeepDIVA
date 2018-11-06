from PIL import Image
import numpy as np


def int_to_one_hot(x, n_classes):
    """
    Read out class encoding from blue channel bit-encoding (1 to [0,0,0,1] -> length determined by the number of classes)
    :param x: integer (pixel value of Blue channel from RGB image)
    :param n_classes: number of class labels
    :return: one-hot encoded list for
    """
    s = '{0:0' + str(n_classes) + 'b}'
    return list(map(int, list(s.format(x))))


def label_img_to_one_hot(np_array):
    """
    TODO: There must be a faster way of doing this
    Convert ground truth label image to one-hot encoded matrix of size image height x image width x #classes
    :param np_array: numpy array of an RGB image
    :return: sparse one-hot encoded class matrix
    """
    im_np = np_array[:, :, 2].astype(np.int8)
    nb_classes = len(int_to_one_hot(im_np.max(), ''))

    class_dict = {x: int_to_one_hot(x, nb_classes) for x in np.unique(im_np)}
    # create the one hot matrix
    one_hot_matrix = np.asanyarray([[class_dict[im_np[i,j]] for j in range(im_np.shape[1])] for i in range(im_np.shape[0])])

    return one_hot_matrix


def one_hot_to_output(matrix, out_path):
    """
    This function converts the one-hot encoded matrix to an image like it was provided in the ground truth
    :param matrix: one-hot encoded matrix
    :param out_path: path to output (e.g. image output/image1.png)
    """
    # create RGB
    matrix = np.char.mod('%d', matrix)
    zeros = (32 - matrix.shape[2]) * '0'
    B = np.array([[int('{}{}'.format(zeros, ''.join(matrix[i][j])), 2) for j in range(matrix.shape[1])] for i in
                  range(matrix.shape[0])])

    RGB = np.dstack((np.zeros(shape=(matrix.shape[0], matrix.shape[1], 2), dtype=np.int8), B))

    # save image
    img = Image.fromarray(RGB.astype('uint8'), 'RGB')
    img.save(out_path)
