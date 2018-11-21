import os

from unittest import TestCase
from PIL import Image
from template.runner.semantic_segmentation.transform_library import transforms

from datasets.image_folder_segmentation import find_classes, is_image_file, ImageFolder


class Test_get_item(TestCase):
    def test_length_of_epoch(self):
        train_dir = "/Users/tomasz/DeepDIVA/datasets/SEGDB/train"
        pages_in_memory = 3
        crops_per_image = 10
        train_ds = ImageFolder(train_dir, pages_in_memory, crops_per_image)
        length_of_epoch = pages_in_memory * crops_per_image * len(train_ds.imgs)
        self.assertEqual(train_ds.__len__(), length_of_epoch)

    def test_get_item(self):
        train_dir = "/Users/tomasz/DeepDIVA/datasets/SEGDB/train"
        pages_in_memory = 3
        crops_per_image = 10
        crop_size = 200
        train_ds = ImageFolder(train_dir, pages_in_memory, crops_per_image, crop_size)

        image_gt_transform = transforms.Compose([
            transforms.RandomTwinCrop()
        ])
        train_ds.transform = image_gt_transform

        index = 0 # value that is not used at all

        length_of_dataset = train_ds.__len__()

        for i in range(length_of_dataset):
            img, gt_one_hot, current_page, current_crop, memory_pass = train_ds.__getitem__(index, test=True)

            # Getting the train dir
            current_memory_pass = os.path.join(train_dir, 'memory_pass_' + str(memory_pass))
            current_page_folder = os.path.join(current_memory_pass, 'page_' + str(current_page))
            current_crop_folder = os.path.join(current_page_folder, 'crop_' + str(current_crop))

            if not os.path.isdir(current_memory_pass):
                os.makedirs(current_memory_pass)
            if not os.path.isdir(current_page_folder):
                os.makedirs(current_page_folder)
            if not os.path.isdir(current_crop_folder):
                os.makedirs(current_crop_folder)

            img.save(current_crop_folder + "/img_" + str(i), "png")
            gt_one_hot.save(current_crop_folder + "/gt_" + str(i), "png")