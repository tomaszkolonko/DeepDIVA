from unittest import TestCase

from datasets.image_folder_segmentation_parser import find_classes_, is_image_file, ImageFolder


class Test_find_classes_(TestCase):
    def test_find_classes_(self):
        array_of_classes = find_classes_()
        self.assertIsNotNone(array_of_classes)
        self.assertTrue(len(array_of_classes) >= 2)

    def test_is_image_file(self):
        image = is_image_file("test.png")
        self.assertTrue(image)
        not_acceptable_format = is_image_file("test.tiff")
        self.assertFalse(not_acceptable_format)

    def test_imageFolder(self):
        path_to_train_data = "/Users/tomasz/DeepDIVA/datasets/SEGDB/train"
        train_ds = ImageFolder(path_to_train_data)
        # check all class variables

        # check root folder string
        self.assertEqual(train_ds.root, path_to_train_data)

        # check img array of tupels
        self.assertIsNotNone(train_ds.imgs)
        array_of_tupels = train_ds.imgs
        for path_to_img, path_to_gt in array_of_tupels:
            self.assertEqual(path_to_img.replace("/img/", "/"), path_to_gt.replace("/gt/", "/"))


        # check if classes are present and more or equal than 2
        self.assertIsNotNone(train_ds.classes)
        self.assertTrue(len(train_ds.classes) >= 2)

        # check if transformations are both set to None
        self.assertIsNone(train_ds.transform)
        self.assertIsNone(train_ds.target_transform)