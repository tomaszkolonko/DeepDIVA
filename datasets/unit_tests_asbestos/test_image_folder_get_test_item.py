import os

from unittest import TestCase

import models
import PIL
from PIL import Image
from template.runner.image_classification.transform_library import transforms

from datasets.image_folder_dataset import find_classes, is_image_file, ImageFolder

# Test::asbestos 9 images: 3 (1024x768) and 6 (1024x1024)
# Test::non-asbestos 9 images 3 (1024x768) and 6 (1024x1024)
# Total 18 images
# exhaustive crops of (224x224) from (1024x768) are 20
# exhaustive crops of (224x224) from (1024x1024) are 25
# length of epoch is 6 * 20 + 12 * 25 = 120 + 250 + 50 = 420

# Crops will be done and saved for visual checking. Every epoch gets its own folder

class Test_get_item(TestCase):
    def setUp(self):
        self.test_dir = "/Users/tomasz/DeepDIVA/datasets/unit_tests_asbestos/ASB_T/test/"
        model_name = "resnet18"

        model_expected_input_size = models.__dict__[model_name]().expected_input_size

        # TODO: needs an implementation of exaustive crop !!!

        self.test_ds = ImageFolder(self.test_dir)

    def test_length_of_epoch(self):
        # length of epochs for the train set is 33
        length_of_epoch = 420
        self.assertEqual(self.test_ds.__len__(), length_of_epoch)

    def test_class_names(self):
        class_name_asbestos = "asbestos"
        class_name_non_asbestos = "non-asbestos"

        self.assertEqual(self.test_ds.classes[0], class_name_asbestos)
        self.assertEqual(self.test_ds.classes[1], class_name_non_asbestos)

    def test_model_expected_input_size(self):
        model_name = "resnet18"
        model_expected_input_size = models.__dict__[model_name]().expected_input_size
        self.assertEqual(model_expected_input_size, (224, 224))