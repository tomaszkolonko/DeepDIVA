import os

from unittest import TestCase
import PIL
from PIL import Image
from template.runner.semantic_segmentation.transform_library import transforms

from datasets.image_folder_segmentation import find_classes, is_image_file, ImageFolder


class Test_get_item(TestCase):
    def setUp(self):
        self.train_dir = "/Users/tomasz/DeepDIVA/datasets/unit_tests_asbestos/ASB_T"

        self.pages_in_memory = 3
        self.crops_per_image = 10
        self.crop_size = 200
        self.train_ds = ImageFolder(self.train_dir, self.pages_in_memory, self.crops_per_image, self.crop_size)

    def test_length_of_epoch(self):
        length_of_epoch = self.pages_in_memory * self.crops_per_image * len(self.train_ds.imgs)
        self.assertEqual(self.train_ds.__len__(), length_of_epoch)

    def test_get_item(self):