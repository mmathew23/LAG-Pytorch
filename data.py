from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import glob
import os
import random

denorm = transforms.Compose([
    transforms.Normalize([0,0,0], [2.0, 2.0, 2.0]),
    transforms.Normalize([-0.5, -0.5, -0.5], [1,1,1]),
    ])


def get_augmentation_transforms(horizontal_flip=True, vertical_flip=False, brightness=0, contrast=0,
        saturation=0, hue=0, color_probability=0.1):
    data_augmentation = []
    if brightness or contrast or saturation or hue:
        data_augmentation.append(transforms.RandomApply((transforms.ColorJitter(brightness, contrast,
            saturation, hue),),  color_probability))
    if horizontal_flip:
        data_augmentation.append(transforms.RandomHorizontalFlip())
    if vertical_flip:
        data_augmentation.append(transforms.RandomVerticalFlip())

    return transforms.Compose(data_augmentation)

#We can use either many images or one large resolution image and sample tiles from it

class FolderDatasetDownsample(Dataset):
    """
    This dataset is created by passing a root folder that contains square shaped images.

    The images will be resized to the size parameter passed into the init
    function and the resulting image is treated as the "label".
    downsample amount is an integer to divide size by. This will result in
    an input image of int(size/downsample) size.
    """
    def __init__(self, root, size=512, downsample=8):
        self.root = root
        self.files = glob.glob(self.root+'/*')
        self.len = len(self.files)
        self.size = size
        #self.resize_full = transforms.Resize((size, size))
        self.resize_full = transforms.CenterCrop(size)
        down_size = int(size/downsample)
        self.resize_down = transforms.Resize((down_size, down_size), Image.BOX)
        self.augmentations = get_augmentation_transforms(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.5,.5,.5], [.5,.5,.5]) #range [-1,1]
            ])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        f = self.files[index]
        img = Image.open(os.path.join(self.root, f)).convert('RGB')
        aug_img = self.resize_full(self.augmentations(img))
        raw_x, raw_y = self.resize_down(aug_img), aug_img
        transformed_x, transformed_y = self.transforms(raw_x), self.transforms(raw_y)
        return transformed_x, transformed_y

    def name(self):
        return 'FolderDatasetDownsample'

class ImageTooSmallError(Exception):
    pass

class LargeImageDataset(Dataset):
    """
    This dataset extracts random tiles from a large image.

    The tiles will be sized to the size parameter passed into the init
    function and the resulting image is treated as the "label".
    downsample amount is an integer to divide size by. This will result in
    an input image of int(size/downsample) size.
    """
    def __init__(self, image_path, size=512, downsample=8):
        self.image_path = image_path
        self.img = Image.open(self.image_path).convert('RGB')
        #len is number of tiles that can be created
        # for WxH image number of tiles is (W-size+1)x(H-size+1)
        w, h = self.img.size
        if w <= size or h <= size:
            raise ImageTooSmallError

        self.len = (w-size+1)*(h-size+1)
        self.cols = (w-size+1)
        self.rows = (h-size+1)
        self.size = size
        self.downsize = int(size/downsample)
        self.tensor_transform = transforms.ToTensor()
        self.augmentations = get_augmentation_transforms(vertical_flip=True,
                brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)
        self.first_transforms = transforms.Compose([
                transforms.ToPILImage(),
                self.augmentations,
            ])
        self.resize_down = transforms.Resize((self.downsize, self.downsize), Image.BOX)
        self.tenorm_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([.5,.5,.5], [.5,.5,.5]) #range [-1,1]
            ])
        self.sres = self.tensor_transform(self.img)

    def indexToTilePos(self, index):
        #we will order like a matrix
        row = index // self.cols
        col = index % self.cols
        return (row, col)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        tilePos = self.indexToTilePos(index)
        row, col = tilePos

        img = self.first_transforms(self.sres[:,row:row+self.size, col:col+self.size])

        lores = self.tenorm_transform(self.resize_down(img))
        return lores, self.tenorm_transform(img)


    def name(self):
        return 'LargeImageDataset'
