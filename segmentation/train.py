import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse

from model.Segmentation import Model
from dataset import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from PIL import ImageFilter
import torchvision.datasets as datasets
import copy

# local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("cuda")

log_path = 'train_log'



class MyTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        # x:173, y:92
        # x:473, y:432
        print('x', x.shape)
        x = x[92:432, 173:473]
        x = self.base_transform(x)
        return x


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_learning_rate(step):
    # if step < 2000:
    #     mul = step / 2000.
    #     return 1e-4 * mul
    # else:
    #     mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
    #     return (1e-4 - 1e-5) * mul + 1e-5

    if step < 100:
        mul = step / 100.
        # print(1e-4 * mul)
        return 1e-4 * mul
    else:
        # return 1e-6
        mul = np.cos((step - 100) / (args.epoch * args.step_per_epoch - 100.) * math.pi) * 0.5 + 0.5
        return (1e-4 - 1e-6) * mul + 1e-6

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)


def visual_feat_visualize(feats):
    # feature_depth = feature_depth.detach().cpu().numpy()
    feat_l = []
    for i in range(feats.shape[0]):
        feature = feats[i]
        depth_min = np.min(feature)
        depth_max = np.max(feature)
        # print('min', np.min(feature_depth))
        # print('max', np.max(feature_depth))
        feature = (feature - depth_min) / (depth_max - depth_min)
        feature = np.uint8(feature * 255)
        feature = cv2.applyColorMap(feature, cv2.COLORMAP_VIRIDIS)
        feat_l.append(feature)
        # cv2.imwrite(save_path, visualize)
    feat_l = np.stack(feat_l, axis=0)
    return feat_l


def train(model, local_rank, start_epoch, args):
    # if local_rank == 0:
    writer = SummaryWriter('train')
    writer_val = SummaryWriter('validate')
    nr_eval = 0
    dataset = HepatomaDataset(args.dataset_path, 'train')
    # sampler = DistributedSampler(dataset)
    # train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True, drop_last=True,
    #                         sampler=sampler)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
    args.step_per_epoch = train_data.__len__()
    step = args.step_per_epoch * start_epoch
    dataset_val = HepatomaDataset(args.dataset_path, 'validation')
    val_data = DataLoader(dataset_val, batch_size=2, pin_memory=True, num_workers=2, shuffle=True)
    print('training...')
    time_stamp = time.time()
    total_epoch = args.epoch
    best_psnr = 0.0
    ce_result_best = 100
    for epoch in range(start_epoch, args.epoch):
        # sampler.set_epoch(epoch)
        for ii, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            for idx in range(len(data)):
                data[idx] = data[idx].to(device)
            img, mask, region_type = data

            # print('segments', torch.max(segments), torch.min(segments))
            # learning_rate = get_learning_rate(step) * args.world_size
            learning_rate = get_learning_rate(step)
            info = model.update(data, learning_rate, training=True, epoch=epoch)  # pass timestep if you are training RIFEm
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            # segments_minus_ = (segments - info['binary_mask']) ** 2
            if step % 100 == 0 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/loss_bce', info['loss_bce'], step)


            if step % 10 == 0 and local_rank == 0:
                img = (img.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')

                mask = (mask.unsqueeze(1).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')

                mask_pre = (
                        info['mask_pre'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(
                    'uint8')
                mask_pre_binary = mask_pre.copy()
                mask_pre_binary[mask_pre_binary>=128] = 255
                mask_pre_binary[mask_pre_binary < 128] = 0

                # print('mask_pre', torch.max(info['mask_pre']), torch.min(info['mask_pre']))

                for i in range(2):
                    # print('imgs', imgs.shape)

                    writer.add_image(str(i) + '/img', img[i], step, dataformats='HWC')
                    writer.add_image(str(i) + '/mask', mask[i], step, dataformats='HWC')
                    writer.add_image(str(i) + '/mask_pre', mask_pre[i], step, dataformats='HWC')
                    writer.add_image(str(i) + '/mask_pre_binary', mask_pre_binary[i], step, dataformats='HWC')


                writer.flush()
            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} lr:{:.4e} '
                      'loss_bce:{:.4e} '
                      .format(epoch, ii, args.step_per_epoch, data_time_interval, train_time_interval, learning_rate,
                              info['loss_bce']
                              ))
            step += 1
            # break
        nr_eval += 1
        # if nr_eval % 5 == 0:
        model.save_model_epoch(log_path, local_rank, epoch)
        ce_result = evaluate(model, val_data, step, local_rank, writer_val, step)
        print('ce_result:', ce_result, 'ce_result_best:' ,ce_result_best)
        if ce_result < ce_result_best:
            ce_result_best = ce_result
            model.save_model_best(log_path, local_rank, epoch)
        model.save_model(log_path, local_rank, epoch)
        # dist.barrier()


def evaluate(model, val_data, nr_eval, local_rank, writer_val, step):

    result_l = []

    for i, data in enumerate(val_data):

        # cout = cout + 1
        for idx in range(len(data)):
            data[idx] = data[idx].to(device)
        img, mask, region_type = data

        with torch.no_grad():
            info = model.update(data, training=False, epoch=epoch)  # pass timestep if you are training RIFEm

        result_l.append(info['loss_bce'])

        img = (img.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')

        mask = (mask.unsqueeze(1).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')

        mask_pre = (
                info['mask_pre'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(
            'uint8')
        mask_pre_binary = mask_pre.copy()
        mask_pre_binary[mask_pre_binary >= 128] = 255
        mask_pre_binary[mask_pre_binary < 128] = 0
        if i == 1:

            for j in range(2):
                # print('imgs', imgs.shape)

                writer_val.add_image(str(j) + '/img', img[j], step, dataformats='HWC')
                writer_val.add_image(str(j) + '/mask', mask[j], step, dataformats='HWC')
                writer_val.add_image(str(j) + '/mask_pre', mask_pre[j], step, dataformats='HWC')
                writer_val.add_image(str(j) + '/mask_pre_binary', mask_pre_binary[j], step, dataformats='HWC')


        writer_val.flush()

    return sum(result_l) / len(result_l)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    parser.add_argument("--resume", action='store_true', default=False)
    parser.add_argument('--dataset_path', default='../detection/data/unmixing/', type=str, help='dataset path')
    parser.add_argument('--split', default=0, type=int, help='split datasets')
    args = parser.parse_args()
    # torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    # args.local_rank = local_rank
    torch.cuda.set_device(args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args.local_rank, epoch=args.epoch)

    epoch = 0
    if args.resume == True:
        # model.load_model_module('train_log')
        state = torch.load('./train_log/flownet_state.pkl')
        epoch = state["epoch"] + 1
        model.load_model_state(state["state_dict"])
        # epoch = 200
    train(model, args.local_rank, epoch, args)

