import os
import torch
from dinov2.data.datasets.image_net import ImageNet


"""  
folder_path = "/home/aleximu/gunes/dinov2/dinov2/data/datasets/fishency/train/n01"

files = os.listdir(folder_path)
image_files = [file for file in files]

for i, file in enumerate(image_files, start=1):
        new_name = f'n01_{i}.jpg'
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)

folder_path = "/home/aleximu/gunes/dinov2/dinov2/data/datasets/fishency/val/n01"

files = os.listdir(folder_path)
image_files = [file for file in files]

for i, file in enumerate(image_files, start=1):
        new_name = f'val_{i}.jpg'
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)

folder_path = "/home/aleximu/gunes/dinov2/dinov2/data/datasets/fishency/test"

files = os.listdir(folder_path)
image_files = [file for file in files]

for i, file in enumerate(image_files, start=1):
        new_name = f'test_{i}.jpg'
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)
  """
"""
for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="/home/aleximu/gunes/dinov2/dinov2/data/datasets/fishency", 
                       extra="/home/aleximu/gunes/dinov2/dinov2/data/datasets/fishency/extra")
    dataset.dump_extra()
"""