import os
import sys
# 获取当前文件的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前目录添加到 sys.path，以便导入 test 模块
sys.path.append(current_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv2
import torchvision.models as models
import torchvision.models.segmentation as segmentation
from backbone import (
    resnet,
    mobilenetv2,
    hrnetv2,
    xception
)
from _deeplab_utils import IntermediateLayerGetter
from _deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3



class Multimodal(nn.Module):

    def __init__(self):
        super(Multimodal, self).__init__()


        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

        backbone = resnet.resnet50(
            pretrained=True,
            replace_stride_with_dilation=replace_stride_with_dilation)

        inplanes = 2048
        low_level_planes = 256

        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, 1, aspp_dilate)

        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.model = DeepLabV3(backbone, classifier)


        self.sigmoid = nn.Sigmoid()

    def forward(self, img, region_type):

        bs, c, h, w, = img.shape
        output = self.model(torch.cat([img, region_type.view(bs, 1, 1, 1).repeat(1, 1, h, w)/4.], dim=1))

        output = self.sigmoid(output)

        return output


