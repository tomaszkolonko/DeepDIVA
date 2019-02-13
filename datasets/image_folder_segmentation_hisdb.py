"""
Load a dataset of historic documents by specifying the folder where its located.
"""

# Utils
import logging
import os
import sys
import math
import os.path
import numpy as np

from datasets.transform_library import transforms, functional

# Torch related stuff
import torch.utils.data as data

from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes():
    classes = ["background", "foreground", "text", "decoration"]
    classes.sort()
    return classes


def make_dataset(directory):
    images = []
    directory = os.path.expanduser(directory)

    # TODO: fix this as soon it is working
    path_imgs = os.path.join(directory, "data")
    path_gts = os.path.join(directory, "gt")

    if not (os.path.isdir(path_imgs) or os.path.isdir(path_gts)):
        logging.error("folder data or gt not found in " + str(directory))

    for _, _, fnames in sorted(os.walk(path_imgs)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path_img = os.path.join(path_imgs, fname)
                fname_gt = fname[:-4] + ".png"
                path_gt = os.path.join(path_gts, fname_gt)
                item = (path_img, path_gt)
                images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'PIL':
        return pil_loader(path)
    else:
        logging.info("Something went wrong with the default_loader in image_folder_segmentation_hisdb.py")
        sys.exit(-1)


def load_dataset(dataset_folder, in_memory=False, workers=1, testing=False, **kwargs):
    """
    Loads the dataset from file system and provides the dataset splits for train validation and test

    The dataset is expected to be in the following structure, where 'dataset_folder' has to point to
    the root of the three folder train/val/test.

    Example:

        dataset_folder = "~/../../data/histDoc"

    which contains the splits sub-folders as follow:

        'dataset_folder'/train
        'dataset_folder'/val
        'dataset_folder'/test

    In each of the three splits (train, val, test) there are two folders. One for the ground truth ("gt")
    and the other for the images ("data"). The ground truth image is of equal size and and encoded the
    following classes: background, foreground, text and decoration.

    Example:

        ../CB55/train/data/page23.png
        ../CB55/train/data/page231.png
        ../CB55/train/gt/page23.png
        ../CB55/train/gt/page231.png

        ../CB55/val/data
        ../CB55/val/gt
        ../CB55/test/data
        ../CB55/test/gt



    Parameters
    ----------
    gt_to_one_hot: function
        Function that converts the ground truth to the one hot encoding

    dataset_folder : string
        Path to the dataset on the file System

    args : dict
        Dictionary of all the CLI arguments passed in

    in_memory : boolean
        Load the whole dataset in memory. If False, only file names are stored and images are loaded
        on demand. This is slower than storing everything in memory.

    workers: int
        Number of workers to use for the dataloaders

    testing: boolean
        Take another path if you are in testing phase

    Returns
    -------
    train_ds : data.Dataset

    val_ds : data.Dataset

    test_ds : data.Dataset
        Train, validation and test splits
    """
    # Get the splits folders
    train_dir = os.path.join(dataset_folder, 'train')
    val_dir = os.path.join(dataset_folder, 'val')
    test_dir = os.path.join(dataset_folder, 'test')

    if in_memory:
        logging.error("With segmentation you don't have the option to put everything into memory")
        sys.exit(-1)

    # Sanity check on the splits folders
    if not os.path.isdir(train_dir):
        logging.error("Train folder not found in the dataset_folder=" + dataset_folder)
        sys.exit(-1)
    if not os.path.isdir(val_dir):
        logging.error("Val folder not found in the dataset_folder=" + dataset_folder)
        sys.exit(-1)
    if not os.path.isdir(test_dir):
        logging.error("Test folder not found in the dataset_folder=" + dataset_folder)
        sys.exit(-1)

    if testing:
        return train_dir, val_dir, test_dir

    # Get an online dataset for each split
    train_ds = ImageFolder(train_dir, **kwargs)
    val_ds = ImageFolder(val_dir, **kwargs)
    test_ds = ImageFolder(test_dir, **kwargs)
    return train_ds, val_ds, test_ds


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/gt/xxx.png
        root/gt/xxy.png
        root/gt/xxz.png

        root/data/xxx.png
        root/data/xxy.png
        root/data/xxz.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    # TODO: transform and target_transform could be the correct places for your cropping
    def __init__(self, root, gt_to_one_hot, num_classes, imgs_in_memory=0, crops_per_image=0, crop_size=10, transform=None, target_transform=None,
                 loader=default_loader, **kwargs):
        classes = find_classes()
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.gt_to_one_hot = gt_to_one_hot
        self.num_classes = num_classes

        # Variables for train and validation sets
        self.imgs_in_memory = imgs_in_memory
        self.crops_per_image = crops_per_image
        self.crop_size = crop_size
        self.current_crop = 0
        self.next_image = 0
        self.memory_position_to_change = 0
        self.memory_pass = 1
        self.test_set = "test" in self.root
        self.images = [None] * imgs_in_memory
        self.gt = [None] * imgs_in_memory

        # Variables for test set
        self.current_page = 0

        if self.test_set:
            self.current_test_image = default_loader(self.imgs[0][0])
            self.current_test_gt = default_loader(self.imgs[0][1])
            self.current_test_gt_name = os.path.basename(self.imgs[0][1])[:-4]
            self.current_test_image_counter = 0
            self.crops_per_test_image = 0
            self.get_item_number = 0

            # TODO: super disgusting... but __len__() needs to know and for that we need
            # TODO: the dimensions of the image
            img = Image.open(open(self.imgs[0][0], 'rb'))
            self.img_heigth = img.size[0]
            self.img_width = img.size[1]
            self.num_vert_crops = math.ceil(img.size[0] / (crop_size/2))
            self.current_vert_crop = 0
            self.num_horiz_crops = math.ceil(img.size[1] / (crop_size/2))
            self.current_horiz_crop = 0
            self.crops_per_image = self.num_vert_crops * self.num_horiz_crops
            # logging.info("*** *** number horizontal_crops: " + str(self.num_horiz_crops))
            # logging.info("*** *** number of vertical_crops: " + str(self.num_vert_crops))
            # logging.info("*** *** length of the dataset: " + str(self.__len__()))


    def __getitem__(self, index, unittesting=False):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, groundtruth) where both are paths to the actual image.
            tuple: for test ((window_input, orig_img_shape, top_left_coordinates_of_crop,
                is_new_img), target)
        """
        # TODO: if you fix the width and height issue, just change the tuple in parameters (for linda)
        if self.test_set:
            # logging.info("*** v_crop: " + str(self.current_vert_crop) + "; h_crop: " + str(self.current_horiz_crop) +
            #              "; of image: " + str(self.current_test_image_counter + 1) +
            #              " of " + str(len(self.imgs)) + " self.current_crop: " + str(self.current_crop))

            # load first image
            if self.current_test_image_counter < len(self.imgs):
                if self.current_vert_crop < self.num_vert_crops:
                    if self.current_horiz_crop < self.num_horiz_crops: # current_horiz_crop < 15
                        if self.current_horiz_crop == (self.num_horiz_crops - 1): # current_horiz_crop == 14
                            output = self.test_crop()
                            self.current_horiz_crop = 0
                            self.current_vert_crop += 1
                            return output
                        else:
                            output = self.test_crop()
                            self.current_horiz_crop += 1
                            return output

                self.load_new_test_data()
                self.reset_counter()
                output = self.test_crop()
                self.current_horiz_crop += 1
                return output


        # Think about moving this initialization to the constructor !!!
        if not self.test_set:
            if self.images[0] is None:
                self.initialize_memory()

            while self.current_page < self.imgs_in_memory:
                while self.current_crop < self.crops_per_image:
                    return self.apply_transformation(unittesting)

                self.update_state_variables()


    def load_new_test_data(self):
        # logging.info("*** loading next image ***")
        self.current_test_image_counter += 1
        self.current_test_image = default_loader(self.imgs[self.current_test_image_counter][0])
        self.current_test_gt = default_loader(self.imgs[self.current_test_image_counter][1])

    def reset_counter(self):
        # logging.info("*** resetting all counters ***")
        self.current_horiz_crop = 0
        self.current_vert_crop = 0

    def test_crop(self):
        """

        :return: (window_input,(original_img_shape), (top_left_coordinates_of_crop),
                    is_new_image, target)
        """
        x_position, y_position = self.get_crop_coordinates()
        # logging.info("x_position: " + str(x_position) + "  //  y_position: " + str(y_position) + "\n")
        window_input_image = functional.crop(self.current_test_image, x_position, y_position, self.crop_size, self.crop_size)
        window_target_image = functional.crop(self.current_test_gt, x_position, y_position, self.crop_size, self.crop_size)

        window_input_torch = functional.to_tensor(window_input_image)
        window_target_torch = functional.to_tensor(window_target_image)

        np_array = (window_target_torch * 255).numpy().astype(np.uint8)
        im_np = np_array[2, :, :].astype(np.uint8)
        #
        # input_np = window_input_torch.numpy()[2,:,:]
        # print(np.unique(im_np))

        one_hot_matrix = self.gt_to_one_hot(window_target_torch, self.num_classes)
        self.current_crop += 1
        return ((window_input_torch, (self.img_width, self.img_heigth), (x_position, y_position),
                 os.path.basename(self.imgs[self.current_test_image_counter][1])[:]), one_hot_matrix)

    def get_crop_coordinates(self):
        if self.current_horiz_crop == (self.num_horiz_crops - 1):
            x_position = self.img_width - self.crop_size
        else:
            x_position = int(self.crop_size/2) * self.current_horiz_crop

        if self.current_vert_crop == (self.num_vert_crops - 1):
            y_position = self.img_heigth - self.crop_size
        else:
            y_position = int(self.crop_size/2) * self.current_vert_crop
        return x_position, y_position


    def apply_transformation(self, unittesting):
        """
        Applies the transformations that have been defined in the setup (setup.py). If no transformations
        have been defined, the PIL image is returned instead.

        :param test:
        :return:
        """
        if self.transform is not None:
            img, gt = self.transform(self.images[self.current_page], self.gt[self.current_page], self.crop_size)
            self.current_crop = self.current_crop + 1
            if unittesting:
                return img, self.gt_to_one_hot(gt, self.num_classes), self.current_page, self.current_crop, self.memory_pass
            else:
                #unique, counts = np.unique(gt.numpy()[2, :, :]*255, return_counts=True)
                #print(dict(zip(unique, counts)))
                return img, self.gt_to_one_hot(gt, self.num_classes)
        else:
            self.current_crop = self.current_crop + 1
            img = self.images[self.current_page]
            gt = self.gt[self.current_page]
            return img, self.gt_to_one_hot(gt, self.num_classes)

    def update_state_variables(self):
        """
        Updates the current_page and the current_crop. If necessary calls update_memory()
        :return:
        """
        self.current_page = self.current_page + 1
        self.current_crop = 0
        if self.current_page == self.imgs_in_memory:
            self.update_memory()
            self.memory_pass = self.memory_pass + 1
            self.current_page = 0

    def __len__(self):
        """
        This function returns the length of an epoch so the dataloader knows when to stop
        :return:
        """
        if self.test_set:
            return self.crops_per_image * len(self.imgs)
        else:
            return len(self.imgs) * self.imgs_in_memory * self.crops_per_image

    def initialize_memory(self):
        """
        First time loading of #pages into memory. If pages_in_memory is set to 3 then the array self.images
        and self.gt will have size of three and be here initialized to the first three images with ground truth.

        :return:
        """
        for i in range(0, self.imgs_in_memory):
            temp_image, temp_gt = self.imgs[i]
            self.next_image = self.next_image + 1

            self.images[i] = self.loader(temp_image)
            self.gt[i] = self.loader(temp_gt)

    def update_memory(self):
        """
        When enough crops have been taken per image from all images residing in memory, the oldest image and
        ground truth will be replaced by a new image and ground truth.

        :return:
        """
        new_image, new_gt = self.imgs[self.next_image % len(self.imgs)]
        self.next_image = (self.next_image + 1) % len(self.imgs)

        self.images[self.memory_position_to_change] = self.loader(new_image)
        self.gt[self.memory_position_to_change] = self.loader(new_gt)
        self.memory_position_to_change = (self.memory_position_to_change + 1) % self.imgs_in_memory


