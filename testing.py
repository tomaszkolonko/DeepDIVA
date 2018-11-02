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


def label_img_to_one_hot(img_path, nb_classes):
    """
    TODO: There must be a faster way of doing this
    Convert ground truth label image to one-hot encoded matrix of size image height x image width x #classes
    :param img_path: path to image
    :param nb_classes: number of class labels
    :return: sparse one-hot encoded class matrix
    """
    im_np = np.array(Image.open(img_path))[:, :, 2].astype(np.int8)

    class_dict = {x: int_to_one_hot(x, nb_classes) for x in np.unique(im_np)}
    # create an empty matrix to fill
    one_hot_matrix = np.ndarray(shape=(im_np.shape[0], im_np.shape[1], nb_classes), dtype=np.int8)

    # fill the matrix
    for key, value in class_dict.items():
        ind = np.where(im_np == key)
        for i, j in zip(ind[0], ind[1]):
            one_hot_matrix[i, j, ] = value

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


def unzip_folders(zip_file, file, folder_extracted, name):
    dirname = os.path.join(folder_extracted, name)
    # unzip files
    zip_file.extract(file, dirname)
    with zipfile.ZipFile(os.path.join(dirname, file), "r") as zip_ref:
        zip_ref.extractall(dirname)
        # delete zips
        os.remove(os.path.join(dirname, file))


def download_HisDB(link, out_folder_name):
    download_path = os.path.join(path_to_save, link.geturl().rsplit('/', 1)[-1])
    # download files
    # urllib.request.urlretrieve(link.geturl(), download_path)
    folder_extracted = os.path.splitext(os.path.join(path_to_save, link.geturl().rsplit('/', 1)[-1].rsplit('.', 1)[0]))[
        0]

    # unpack relevant folders
    zip_file = zipfile.ZipFile(download_path)

    # unpack imgs and gt
    for file in zip_file.namelist():
        if file.startswith('img'):
            unzip_folders(zip_file, file, folder_extracted, 'imgs')
        elif file.startswith('pixel'):
            unzip_folders(zip_file, file, folder_extracted, 'gt')

    for subdir, dirs, files in os.walk(folder_extracted):
        for file in files:
            shutil.move(os.path.join(subdir, file), os.path.join(os.path.dirname(subdir), file))

    if out_folder_name == 'train':
        train_folder = os.path.join(os.path.dirname(folder_extracted), out_folder_name)

    os.rename(folder_extracted, os.path.join(os.path.dirname(folder_extracted), out_folder_name))


def hisDB(path_to_save):
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
    # links to HisDB
    linktrain_all = urllib.parse.urlparse(
        'https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/hisdoc/all.zip')
    linktest_all = urllib.parse.urlparse(
        'https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/hisdoc/private-test/all-privateTest.zip')

    download_HisDB(linktest_all, 'test')
    download_HisDB(linktrain_all, 'train')


path_to_save = '/home/linda/Documents/PhD/datasets/'
hisDB(path_to_save)

