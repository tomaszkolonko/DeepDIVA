# Torch related stuff
from PIL import Image
import torchvision.transforms as transforms

# Utils
import logging
import os
import sys

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def is_file_name_image(file_name):
    file_name = file_name.lower()
    return any(file_name.endswith(extension) for extension in IMG_EXTENSIONS)

def get_file_names(directory):
    images = []
    path_images = os.path.expanduser(directory)

    for _, _, file_names in sorted(os.walk(path_images)):
        for file_name in sorted(file_names):
            if is_file_name_image(file_name):
                path_image = os.path.join(path_images, file_name)
                images.append(path_image)
    return images

def load_image(path_img):
    with open(path_img, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB'), path_img

def expand_and_save_image(img):
    for image in img:
        transform = transforms.Resize(224)
        new_image = transform.__call__(image[0])
        new_image.save(image[1])


array_of_images = get_file_names("/Users/tomasz/DeepDIVA/datasets/asbestos_resizing")
tuple_of_images = []
for a in array_of_images:
    tuple_of_images.append(load_image(a))

expand_and_save_image(tuple_of_images)