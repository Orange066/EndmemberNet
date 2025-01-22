import os
import cv2
import numpy as np
from PIL import Image
import time
from skimage import io
from collections import defaultdict

def data_augment(tif_files):

    data_augment_l=[]

    img_l = []
    for idx, file in enumerate(tif_files):
        img = io.imread(file)
        img_array = np.array(img)
        img_l.append(img_array)

    img_ori = np.stack(img_l, axis=2)
    img_ori = np.max(img_ori, axis=2)
    img_ori = (img_ori - img_ori.min()) / (img_ori.max() - img_ori.min()) * 255

    img_ori_copy=img_ori.copy()
    data_augment_l.append(img_ori)

    image = img_ori.astype(np.uint8)
    image_copy = image.copy()
    num_min = np.min(image) - 1
    num_max = np.max(image) + 1
    hist, bin_edges = np.histogram(image, bins=20, range=(num_min, num_max))

    hist_max = np.max(hist)
    delete_idx = 0
    for h_idx, h in enumerate(hist):
        if h > hist_max // 2:
            delete_idx = h_idx

    delete_hist = bin_edges[delete_idx + 1]
    image_copy[image_copy < delete_hist] = 0
    image_copy[image_copy >= delete_hist] = 1

    # 对于灰度图像，直接应用均衡化
    image = image * image_copy
    result_image = cv2.equalizeHist(image)
    data_augment_l.append(result_image)

    select_l = []
    for loc_0 in range(5):
        for loc_1 in range(loc_0+1, 5):
            select_l_tmp = [0, 0, 0, 0, 0]
            select_l_tmp[loc_1] = 1
            select_l_tmp[loc_0] = 1
            select_l.append(select_l_tmp)

    for loc_0 in range(5):
        select_l_tmp = [0, 0, 0, 0, 0]
        select_l_tmp[loc_0] = 1
        select_l.append(select_l_tmp)

    # for loc_0 in range(5):
    #     select_l_tmp = [1, 1, 1, 1, 1]
    #     select_l_tmp[loc_0] = 0
    #     select_l.append(select_l_tmp)
    #
    # for loc_0 in range(5):
    #     select_l_tmp = [1, 1, 1, 1, 1]
    #     select_l.append(select_l_tmp)

    for j in range(len(select_l)):
        # 生成五个随机权重
        weights = np.random.rand(5)
        weights[0] = weights[0] * select_l[j][0]
        weights[1] = weights[1] * select_l[j][1]
        weights[2] = weights[2] * select_l[j][2]
        weights[3] = weights[3] * select_l[j][3]
        weights[4] = weights[4] * select_l[j][4]

        # 使用权重对数组进行加权求和
        weighted_sum = np.zeros_like(img_l[0])
        for arr, weight in zip(img_l, weights):
            weighted_sum += arr * weight
        # weighted_sum = img_l[j]
        img_array = (weighted_sum - weighted_sum.min()) / (weighted_sum.max() - weighted_sum.min() + 1e-4) * 255
        data_augment_l.append(img_array)

        image = img_array.astype(np.uint8)
        image_copy = image.copy()
        num_min = np.min(image) - 1
        num_max = np.max(image) + 1
        hist, bin_edges = np.histogram(image, bins=20, range=(num_min, num_max))

        hist_max = np.max(hist)
        delete_idx = 0
        for h_idx, h in enumerate(hist):
            if h > hist_max // 2:
                delete_idx = h_idx

        delete_hist = bin_edges[delete_idx + 1]
        image_copy[image_copy < delete_hist] = 0
        image_copy[image_copy >= delete_hist] = 1

        # 对于灰度图像，直接应用均衡化
        image = image * image_copy
        result_image = cv2.equalizeHist(image)
        data_augment_l.append(result_image)

    return data_augment_l, img_ori_copy, img_l


def invert_convert(size, norm_coords):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x, y, w, h = norm_coords

    # 将x, y, w, h转换回原始图像尺寸的比例
    x = (x / dw) + 1
    y = (y / dh) + 1
    w = w / dw
    h = h / dh

    # 计算原始坐标框
    x_min = x - (w / 2.0)
    x_max = x + (w / 2.0)
    y_min = y - (h / 2.0)
    y_max = y + (h / 2.0)

    return (x_min, y_min, x_max, y_max)


def process_boxs(pred_boxs, width, height):
    results = []
    num_classes = 5
    for cls in range(num_classes):
        cls_boxes = pred_boxs[:, 5] == cls

        if np.any(cls_boxes):
            max_idx = np.argmax(pred_boxs[cls_boxes, 4])
            x, y, w, h, confident = pred_boxs[cls_boxes][max_idx][:5]
            # x_min, y_min, x_max, y_max = invert_convert((width, height), (float(x), float(y), float(w), float(h)))
            # results.append([confident, x_min, y_min, x_max, y_max, cls])
            results.append([confident, x, y, w, h, cls])
        # else:
        #     results.append([-1, 0, 0, 0, 0, cls])

    # 将结果转换为NumPy数组，如果需要的话
    if len(results) == 5:
        results_array = np.array(results)
        return results_array
    return None



