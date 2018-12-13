import os

from unittest import TestCase

import models
import PIL
from PIL import Image
from template.runner.image_classification.transform_library import transforms

from datasets.image_folder_dataset import find_classes, is_image_file, ImageFolder

# Train::asbestos 19 images 4 (1024x768) and 15 (1024x1024)
# Train::non-asbestos 14 images 5 (1024x768) and 9 (1024x1024)
# Total 33 images, which also is the length of the epoch

# Crops will be done and saved for visual checking. Every epoch gets its own folder

class Test_get_item(TestCase):
    def setUp(self):
        self.train_dir = "/Users/tomasz/DeepDIVA/datasets/unit_tests_asbestos/ASB_T/train/"
        model_name = "resnet18"

        model_expected_input_size = models.__dict__[model_name]().expected_input_size

        transform = transforms.Compose([
            transforms.RandomRotation((0, 360)),
            transforms.RandomCrop(model_expected_input_size)
        ])

        self.train_ds = ImageFolder(self.train_dir, transform)

    def test_length_of_epoch(self):
        # length of epochs for the train set is 33
        length_of_epoch = len(self.train_ds.imgs)
        self.assertEqual(self.train_ds.__len__(), length_of_epoch)

    def test_class_names(self):
        class_name_asbestos = "asbestos"
        class_name_non_asbestos = "non-asbestos"

        self.assertEqual(self.train_ds.classes[0], class_name_asbestos)
        self.assertEqual(self.train_ds.classes[1], class_name_non_asbestos)

    def test_model_expected_input_size(self):
        model_name = "resnet18"
        model_expected_input_size = models.__dict__[model_name]().expected_input_size
        self.assertEqual(model_expected_input_size, (224, 224))

    def test_get_item(self):

        index = 0 # value that is not used at all

        length_of_dataset = self.train_ds.__len__()

        for i in range(length_of_dataset):
            img, target = self.train_ds.__getitem__(index)

            # Getting the train dir
            current_memory_pass = os.path.join(self.train_dir, 'memory_pass_' + str(memory_pass))
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