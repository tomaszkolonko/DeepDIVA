from unittest import TestCase

from datasets.image_folder_segmentation_hisdb import load_dataset


class Test_load_dataset(TestCase):
    def test_load_dataset(self):
        path_to_data = "/Users/tomasz/DeepDIVA/datasets/SEGDB"

        train_dir, val_dir, test_dir = load_dataset(path_to_data, testing=True)
        self.assertEqual(train_dir, path_to_data + "/train")
        self.assertEqual(val_dir, path_to_data + "/val")
        self.assertEqual(test_dir, path_to_data + "/test")


