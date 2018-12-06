import os

from unittest import TestCase
import PIL
from PIL import Image
from template.runner.semantic_segmentation.transform_library import transforms

from datasets.image_folder_segmentation import find_classes, is_image_file, ImageFolder


class Test_get_item(TestCase):
    def setUp(self):
        self.test_dir = "/Users/tomasz/DeepDIVA/datasets/unit_tsts/test"
        self.pages_in_memory = 3
        self.crops_per_image = 10
        self.crop_size = 200
        self.test_ds = ImageFolder(self.test_dir, self.pages_in_memory, self.crops_per_image, self.crop_size)

    def test_length_of_epoch(self):
        # check init to see the formula how it was calculated
        # vertical crops: 25
        # horizontal crops: 33
        # number of test images: 5
        self.assertEqual(self.test_ds.__len__(), 25*33*5)

    def test_get_item(self):
        image_gt_transform = transforms.Compose([
            transforms.RandomTwinCrop(),
            transforms.ToTensorTwinImage()
        ])
        self.test_ds.transform = image_gt_transform

        index = 0 # value that is not used at all

        length_of_dataset = self.test_ds.__len__()

        for i in range(length_of_dataset):
            img, gt_one_hot, current_page, current_crop, memory_pass = self.test_ds.__getitem__(index, unittesting=True)

            # Getting the train dir
            current_memory_pass = os.path.join(self.test_ds, 'memory_pass_' + str(memory_pass))
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


