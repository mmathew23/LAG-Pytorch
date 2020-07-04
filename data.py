from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import glob
import os
import random

#We can use either many images or one large resolution image and sample tiles from it

class FolderDatasetDownsample(Dataset):
        """
        This dataset is created by passing a root folder that contains square shaped images.

        The images will be resized to the size parameter passed into the init
        function and the resulting image is treated as the "label".
        downsample amount is an integer to divide size by. This will result in
        an input image of int(size/downsample) size.
        shuffle will pick a random image from the dataset with replacement.
        """
    def __init__(self, root, size=512, downsample=8, shuffle=True):
        self.root = root
        self.files = glob.glob(self.root+'/*')
        self.len = len(self.files)
        self.shuffle = shuffle
        self.size = size
        self.resize_full = transforms.Resize(size, size)
        down_size = int(size/downsample)
        self.resize_down = transforms.Resize(down_size, down_size)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #normalize to imagenet stats maybe change
            ])

    def __len(self):
        return self.len

    def __getitem__(self, index):
        if self.shuffle:
            index = random.randint(0, self.len-1)
        f = self.files[index]
        img = Image.open(os.path.join(self.root, f)).convert('RGB')
        raw_x, raw_y = self.resize_down(img), self.resize_full(img)
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
    shuffle will pick a random tile from the image with replacement.
    """
    def __init__(self, image_path, size=512, downsample=8, shuffle=True):
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
        self.shuffle = shuffle
        self.size = size
        self.downsize = int(size/downsample)
        down_size_h = int(h/downsample)
        down_size_w = int(w/downsample)
        self.resize_down = transforms.Resize((down_size_h, down_size_w), PIL.Image.BICUBIC)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #normalize to imagenet stats maybe change
            ])
        self.lores = self.transforms(self.resize_down(self.img))
        self.sres = self.transforms(self.img)

    def indexToTilePos(self, index):
        #we will order like a matrix
        row = index // self.rows
        col = index % self.cols
        return (row, col)

    def __len(self):
        return self.len

    def __getitem__(self, index):
        if self.shuffle:
            index = random.randint(0, self.len-1)
        tilePos = self.indexToTilePos(index)
        row, col = tilePos
        return self.lores[:,row:row+self.downsize, col:col+self.downsize], self.sres[:,row:row+self.size, col:col+self.size]


    def name(self):
        return 'LargeImageDataset'
