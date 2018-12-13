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
        epochs = 2

        length_of_dataset = self.train_ds.__len__()

        for i in range(epochs):
            current_epoch_path_name = self.create_folder(i)
            for j in range(length_of_dataset):
                img, target, path = self.train_ds.__getitem__(j)
                image_base_name = os.path.basename(path)


                img.save(current_epoch_path_name + "/" + image_base_name, "png")


    def create_folder(self, i):
        current_epoch_pass = os.path.join(self.train_dir, 'epoch_' + str(i))
        if not os.path.isdir(current_epoch_pass):
            os.makedirs(current_epoch_pass)
        return current_epoch_pass