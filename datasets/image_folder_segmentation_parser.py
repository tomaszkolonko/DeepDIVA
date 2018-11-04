import torch.utils.data as data

from PIL import Image
import os
import sys
import os.path
import logging

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


# def find_classes(dir):
#     # TODO: think about how not to hardcode this array
#     classes = ["background", "foreground", "text", "decoration"]
#     classes.sort()
#     # TODO: really necessary? Probably not...
#     class_to_idx = {classes[i]: i for i in range(len(classes))}
#     return classes, class_to_idx

def find_classes_():
    classes = ["background", "foreground", "text", "decoration"]
    classes.sort()
    return classes


# def make_dataset(dir, class_to_idx):
#     images = []
#     dir = os.path.expanduser(dir)
#     for target in sorted(os.listdir(dir)):
#         d = os.path.join(dir, target)
#         if not os.path.isdir(d):
#             continue
#
#         for root, _, fnames in sorted(os.walk(d)):
#             for fname in sorted(fnames):
#                 if is_image_file(fname):
#                     path = os.path.join(root, fname)
#                     item = (path, class_to_idx[target])
#                     images.append(item)
#
#     return images


def make_dataset_(dir):
    images = []
    dir = os.path.expanduser(dir)

    # TODO: fix this as soon it is working
    path_imgs = os.path.join(dir, "img")
    path_gts = os.path.join(dir, "gt")

    if not (os.path.isdir(path_imgs) or os.path.isdir(path_gts)):
        logging.error("folder img or gt not found in " + str(dir))


    for _, _, fnames in sorted(os.walk(path_imgs)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path_img = os.path.join(path_imgs, fname)
                path_gt = os.path.join(path_gts, fname)
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
        pil_loader(path)
    else:
        logging.info("Something went wrong with the default_loader in image_folder_segmentation_parser.py")
        sys.exit(-1)


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/gt/xxx.png
        root/gt/xxy.png
        root/gt/xxz.png

        root/img/xxx.png
        root/img/xxy.png
        root/img/xxz.png

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
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        # classes, class_to_idx = find_classes(root)
        classes = find_classes_()
        imgs = make_dataset_(root) #, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        # self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, groundtruth) where both are paths to the actual image.
        """
        path_img, path_gt = self.imgs[index]
        img = self.loader(path_img)
        gt = self.loader(path_gt)
        if self.transform is not None:
            img = self.transform(img)
            gt = self.transform(gt)
        # TODO: unclear if target_transform is still needed for segmentation purpose
        # if self.target_transform is not None:
        #     img = self.transform(img)
        #     gt = self.transform(gt)

        return img, gt

    def __len__(self):
        return len(self.imgs)
