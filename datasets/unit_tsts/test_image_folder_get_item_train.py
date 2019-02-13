import os

from unittest import TestCase
import PIL
from PIL import Image
from datasets.transform_library import transforms

from datasets.image_folder_segmentation_hisdb import find_classes, is_image_file, ImageFolder


class Test_get_item(TestCase):
    def setUp(self):
        self.train_dir = "/Users/tomasz/DeepDIVA/datasets/unit_tsts/train"
        self.pages_in_memory = 3
        self.crops_per_image = 10
        self.crop_size = 200
        self.train_ds = ImageFolder(self.train_dir, self.pages_in_memory, self.crops_per_image, self.crop_size)

    def test_length_of_epoch(self):
        length_of_epoch = self.pages_in_memory * self.crops_per_image * len(self.train_ds.imgs)
        self.assertEqual(self.train_ds.__len__(), length_of_epoch)

    def test_get_item(self):
        image_gt_transform = transforms.Compose([
            transforms.RandomTwinCrop()
        ])
        self.train_ds.transform = image_gt_transform

        index = 0 # value that is not used at all

        length_of_dataset = self.train_ds.__len__()

        for i in range(length_of_dataset):
            img, gt_one_hot, current_page, current_crop, memory_pass = self.train_ds.__getitem__(index, unittesting=True)

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

    def test_initialize_memory(self):
        self.train_ds.initialize_memory()
        self.assertEqual(len(self.train_ds.images), self.pages_in_memory)
        self.assertEqual(len(self.train_ds.images), len(self.train_ds.gt))
        self.assertIsInstance(self.train_ds.images, type([PIL.Image.Image, PIL.Image.Image, PIL.Image.Image]))
        self.assertIsInstance(self.train_ds.gt, type([PIL.Image.Image, PIL.Image.Image, PIL.Image.Image]))
        self.assertEqual(self.train_ds.next_image, 3)
        self.assertEqual(len(self.train_ds.imgs), 5)

    def test_update_memory(self):
        self.train_ds.initialize_memory()

        # check visually with print statements.

        list_of_previous_images = [hex(id(x)) for x in self.train_ds.images]
        list_of_previous_gt = [hex(id(x)) for x in self.train_ds.gt]

        for i in range(len(self.train_ds.imgs)):
            self.train_ds.update_memory()

            list_of_current_images = [hex(id(x)) for x in self.train_ds.images]
            list_of_current_gt = [hex(id(x)) for x in self.train_ds.gt]

            # First element in imgages and gt get's changed
            self.assertNotEqual(list_of_previous_images[i % self.pages_in_memory],
                                list_of_current_images[i % self.pages_in_memory])
            self.assertEqual(list_of_previous_images[(i+1) % self.pages_in_memory],
                             list_of_current_images[(i+1) % self.pages_in_memory])
            self.assertEqual(list_of_previous_images[(i+2) % self.pages_in_memory],
                             list_of_current_images[(i+2) % self.pages_in_memory])

            self.assertNotEqual(list_of_previous_gt[i % self.pages_in_memory],
                                list_of_current_gt[i % self.pages_in_memory])
            self.assertEqual(list_of_previous_gt[(i + 1) % self.pages_in_memory],
                             list_of_current_gt[(i + 1) % self.pages_in_memory])
            self.assertEqual(list_of_previous_gt[(i + 2) % self.pages_in_memory],
                             list_of_current_gt[(i + 2) % self.pages_in_memory])

            list_of_previous_images = list_of_current_images
            list_of_previous_gt = list_of_current_gt

