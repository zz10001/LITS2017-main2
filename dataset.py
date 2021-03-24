import numpy as np
import cv2 #https://www.jianshu.com/p/f2e88197e81d
import random

from skimage.io import imread
from skimage import color

import torch
import torch.utils.data
from torchvision import datasets, models, transforms
import argparse

class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, aug=False):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = list(map(lambda x: x.replace('volume', 'segmentation').replace('image','mask'), self.img_paths))
        # self.mask_paths = mask_paths
        self.aug = aug

        # print(self.img_paths,self.mask_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        npimage = np.load(img_path)
        npmask = np.load(mask_path)

        npimage = npimage[:, :, :, np.newaxis]
        npimage = npimage.transpose((3, 0, 1, 2))

        npmask = npmask[:, :, :, np.newaxis]
        npmask = npmask.transpose((3, 0, 1, 2))

        npmask = npmask.astype("float32")
        npimage = npimage.astype("float32")
        #print(npimage.shape)

        return npimage,npmask


       