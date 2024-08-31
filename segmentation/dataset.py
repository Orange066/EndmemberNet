import os
import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
import lmdb
import re
import math
from skimage import io

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from copy import deepcopy
from typing import Tuple
from torch.nn.functional import threshold, normalize
import torch.nn.functional as F


class HepatomaDataset(Dataset):
    def __init__(self, path, stage):

        self.path = path
        self.stage = stage

        if self.stage == 'train':
            files = os.listdir(os.path.join(path, 'AllData_ori_patch_resize', 'train'))
            masks = os.listdir(os.path.join(path, 'AllMask_patch_resize', 'train'))
            files.sort()
            masks.sort()
            self.files = []
            self.masks = []
            for file in files:
                self.files.append(os.path.join(path, 'AllData_ori_patch_resize', 'train', file))
            for mask in masks:
                self.masks.append(os.path.join(path, 'AllMask_patch_resize', 'train', mask))
        elif self.stage == 'validation':
            files = os.listdir(os.path.join(path, 'AllData_ori_patch_resize', 'valid'))
            masks = os.listdir(os.path.join(path, 'AllMask_patch_resize', 'valid'))
            files.sort()
            masks.sort()
            self.files = []
            self.masks = []
            for file in files:
                self.files.append(os.path.join(path, 'AllData_ori_patch_resize', 'valid', file))
            for mask in masks:
                self.masks.append(os.path.join(path, 'AllMask_patch_resize', 'valid', mask))
        elif self.stage == 'test':
            files = os.listdir(os.path.join(path, 'AllData_ori_patch_resize', 'test'))
            masks = os.listdir(os.path.join(path, 'AllMask_patch_resize', 'test'))
            files.sort()
            masks.sort()
            self.files = []
            self.masks = []
            for file in files:
                self.files.append(os.path.join(path, 'AllData_ori_patch_resize', 'test', file))
            for mask in masks:
                self.masks.append(os.path.join(path, 'AllMask_patch_resize', 'test', mask))



        # print('stage', stage)
        # print('self.files', len(self.files))
        # print('self.masks', len(self.masks))


    def __len__(self):
        return len(self.files)


    def random_perspective(self,
                           im,
                           mask,
                           degrees=10,
                           translate=.1,
                           scale=.1,
                           shear=10,
                           perspective=0.0,
                           border=(0, 0)):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # targets = [cls, xyxy]

        height = im.shape[0] + border[0] * 2  # shape(h,w,c)
        width = im.shape[1] + border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1 + scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * width)  # x translation (pixels)
        T[1, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * height)  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                # im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
                # im_mask = cv2.warpPerspective(im_mask, M, dsize=(width, height), borderValue=(114, 114, 114))
                im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(0, 0, 0))
                mask = cv2.warpPerspective(mask, M, dsize=(width, height), borderValue=(0, 0, 0))
            else:  # affine
                # im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
                # im_mask = cv2.warpAffine(im_mask, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
                im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(0, 0, 0))
                mask = cv2.warpAffine(mask, M[:2], dsize=(width, height), borderValue=(0, 0, 0))



        return im, mask


    def augment_flip(self, img, mask):


        if random.uniform(0, 1) < 0.5:
            # print('here1')
            img = img[:, ::-1]

            mask = mask[:, ::-1]

        if random.uniform(0, 1) < 0.5:
            img = img[::-1, :]
            mask = mask[::-1, :]

        p = random.uniform(0, 1)

        if p < 0.25:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        elif p < 0.5:
            img = cv2.rotate(img, cv2.ROTATE_180)
            mask = cv2.rotate(mask, cv2.ROTATE_180)
        elif p < 0.75:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return img, mask

    def __getitem__(self, index):

        img = cv2.imread(self.files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)

        if self.stage == 'train':
            img, mask = self.augment_flip(img, mask)
            if random.uniform(0, 1) < 0.2:
                img, mask = self.random_perspective(img, mask)


        img = torch.from_numpy(img.copy()).permute(2, 0, 1) / 255.
        mask = torch.from_numpy(mask.copy()) / 255.

        if '-tumor' in self.files[index]:
            region_type = 0
        elif '-intestine' in self.files[index]:
            region_type = 1
        elif '-colon' in self.files[index]:
            region_type = 2
        elif '-mln' in self.files[index]:
            region_type = 3
        elif '-vessel' in self.files[index]:
            region_type = 4


        region_type = torch.ones(1) * region_type

        return img, mask, region_type
