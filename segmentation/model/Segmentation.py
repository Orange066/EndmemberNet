import os
import sys
# 获取当前文件的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前目录添加到 sys.path，以便导入 test 模块
sys.path.append(current_dir)


import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from Multimodal import *

import math

device = torch.device("cuda")
# device = torch.device("cpu")
import cv2 as cv2

import time


class Model:
    def __init__(self, local_rank=-1, arbitrary=False, epoch=None, count_time=False):

        self.multimodal = Multimodal()

        self.device(device)
        self.optimG = AdamW(self.multimodal.parameters(), lr=1e-6,
                            weight_decay=1e-3)  # use large weight decay may avoid NaN loss
        self.bce_loss = nn.BCELoss().to(device)

    def load_model_u(self, rank=0):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        if rank <= 0:
            self.flownet_u.load_state_dict(convert(torch.load('/home/user3/ECCV2022-RIFE-21/train_log/flownet.pkl')))

    def train(self):
        self.multimodal.train()

    def eval(self):
        self.multimodal.eval()

    def device(self,d):
        self.multimodal.to(d)

    def load_model(self, path, rank=0):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        if rank <= 0:
            # self.multimodal.load_state_dict(convert(torch.load('{}/multimodal.pkl'.format(path), map_location=device)))
            self.multimodal.load_state_dict(torch.load('{}/multimodal.pkl'.format(path), map_location=device))

    def load_model_epoch(self, path, epoch, rank=0):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        if rank <= 0:
            self.multimodal.load_state_dict(
                convert(torch.load('{}/multimodal_{}.pkl'.format(path, epoch), map_location=device)))

    def load_model_best(self, path, rank=0):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        if rank <= 0:
            self.multimodal.load_state_dict(convert(torch.load('{}/multimodal_best.pkl'.format(path), map_location=device)))

    def load_model_module(self, path, rank=0):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        if rank <= 0:
            self.multimodal.load_state_dict(torch.load('{}/multimodal.pkl'.format(path)))

    def load_model_state(self, state, rank=0):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        if rank <= 0:
            self.multimodal.load_state_dict(state)

    def save_model(self, path, rank=0, epoch=0):
        if rank == 0:
            torch.save(self.multimodal.state_dict(), '{}/multimodal.pkl'.format(path))
            state_dict = {
                "epoch": epoch,
                "state_dict": self.multimodal.state_dict(),
            }
            torch.save(state_dict, '{}/multimodal_state.pkl'.format(path))

    def save_model_best(self, path, rank=0, epoch=0):
        if rank == 0:
            torch.save(self.multimodal.state_dict(), '{}/multimodal_best.pkl'.format(path))
            state_dict = {
                "epoch": epoch,
                "state_dict": self.multimodal.state_dict(),
            }
            torch.save(state_dict, '{}/multimodal_state_best.pkl'.format(path))

    def save_model_epoch(self, path, rank=0, epoch=0):
        if rank == 0:
            torch.save(self.multimodal.state_dict(), '{}/multimodal_{}.pkl'.format(path, epoch))
            state_dict = {
                "epoch": epoch,
                "state_dict": self.multimodal.state_dict(),
            }
            torch.save(state_dict, '{}/multimodal_state_{}.pkl'.format(path, epoch))

    def save_model_pt11(self):
        torch.save(self.multimodal.state_dict(), './train_log/multimodal_11.pkl', _use_new_zipfile_serialization=False)

    def load_model_pt11(self, path, rank=0):
        self.multimodal.load_state_dict(torch.load('{}/multimodal_11.pkl'.format(path), map_location=device))

    def get_channel_mask(self):
        return self.multimodal.get_channel_mask()

    def inference(self, img, region_type):
        start_time = time.time()
        output = self.multimodal(img, region_type)
        end_time = time.time()
        return end_time - start_time, output

    def test_forward(self, x):

        binary_mask_liver, _ = self.multimodal(x)

        return binary_mask_liver

    def pad(self, im, pad_width):
        h, w = im.shape[-2:]
        mh = h % pad_width
        ph = 0 if mh == 0 else pad_width - mh
        mw = w % pad_width
        pw = 0 if mw == 0 else pad_width - mw
        shape = [s for s in im.shape]
        shape[-2] += ph
        shape[-1] += pw
        im_p = torch.zeros(shape).float()
        im_p = im_p.to(im.device)
        im_p[..., :h, :w] = im
        im = im_p
        return im

    def get_robust_weight(self, flow_pred, flow_gt, beta):
        epe = ((flow_pred.detach() - flow_gt) ** 2).sum(dim=1, keepdim=True) ** 0.5
        robust_weight = torch.exp(-beta * epe)
        return robust_weight

    def update(self, data, learning_rate=0, mul=1, training=True, epoch=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate

        img, mask, region_type = data


        if training:
            self.train()
        else:
            self.eval()

        output = self.multimodal(img,region_type)

        loss_bce = self.bce_loss(output[:, 0], mask)
        if training:

            self.optimG.zero_grad()

            loss_G = loss_bce
            loss_G.backward()

            self.optimG.step()
        return {
            'mask_pre': output,
            'loss_bce': loss_bce,
        }
