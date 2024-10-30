import gradio as gr
import numpy as np
import random
import imageio
from tifffile import imread, imsave

import os, cv2
from tqdm import tqdm
import torch
import utility
from PIL import Image
from skimage import io

# yolo-
import argparse
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT_BASE = FILE.parents[0]

ROOT = os.path.join(ROOT_BASE, 'segmentation')
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# print(str(ROOT))
ROOT = os.path.join(ROOT_BASE, 'detection')
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# print(str(ROOT))
# exit(0)
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from detection.models.common import DetectMultiBackend
from detection.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements,
                                         colorstr, cv2,
                                         increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer,
                                         xyxy2xywh)
from detection.utils.torch_utils import select_device, smart_inference_mode

from util_detection import *
from torch.nn import functional as F
from segmentation.model.Segmentation import Model
import zipfile
import time
import pandas as pd
import threading


DEVICES = ['CPU', 'CUDA']
QUANT = ['float32', 'float16', ]
TASKS = ['Image', 'Video']
INPUTS = ['SR', 'Denoising', 'Isotropic', 'Projection', 'Volumetric']
MODEL_DET = None
MODEL_SEG = None
ARGS = None


class Args:
    model = 'SwinIR'
    test_only = True
    resume = 0
    modelpath = None
    save = None
    task = None
    dir_data = None
    dir_demo = None
    data_test = None

    epoch = 1000
    batch_size = 16
    patch_size = None
    rgb_range = 1
    n_colors = 1
    inch = None
    datamin = 0
    datamax = 100

    cpu = False
    print_every = 1000
    test_every = 2000
    load = ''
    lr = 0.00005
    n_GPUs = 1
    n_resblocks = 8
    n_feats = 32
    save_models = True
    save_results = True
    save_gt = False

    debug = False
    scale = None
    chunk_size = 144
    n_hashes = 4
    chop = False
    self_ensemble = False
    no_augment = False
    inputchannel = None

    act = 'relu'
    extend = '.'
    res_scale = 0.1
    shift_mean = True
    dilation = False
    precision = 'single'

    seed = 1
    local_rank = 0
    n_threads = 0
    reset = False
    split_batch = 1
    gan_k = 1


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='checkpoints/detection/best.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.05, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def load_model(preprocess_type, device, progress=gr.Progress()):
    global MODEL_DET, ARGS, MODEL_SEG

    ARGS = parse_opt()

    if preprocess_type == 'Image':
        ARGS.preprocess_type = 'Image'
    elif 'Video' in preprocess_type:
        ARGS.preprocess_type = 'Video'
    else:
        gr.Error("Preprocess type not found!")
        return "Preprocess type not found"

    if device == 'CUDA':
        ARGS.device = '0'
    elif device == 'CPU':
        ARGS.device = 'cpu'
    else:
        gr.Error("Device not found!")
        return "Device not found"

    # ARGS.source = input_message

    if MODEL_DET is not None:
        del MODEL_DET

    if MODEL_SEG is not None:
        del MODEL_SEG

    # Load model
    device = select_device(ARGS.device)
    print('ARGS.weights:', ARGS.weights)
    MODEL_DET = DetectMultiBackend(ARGS.weights, device=device, dnn=ARGS.dnn, data=ARGS.data, fp16=ARGS.half)
    MODEL_DET.eval()
    MODEL_SEG = Model()
    MODEL_SEG.load_model('checkpoints/segmentation')
    MODEL_SEG.eval()
    MODEL_SEG.device(device)
    # stride, names, pt = model.stride, model.names, model.pt
    # imgsz = check_img_size(ARGS.imgsz, s=stride)  # check image size
    # print(model)

    return 'Model loaded on %s' % (device)


def data_generate_enhance(tif_files):
    data_augment_l, data_ori_rgb, data_ori_tif_l = data_augment(tif_files)
    return data_augment_l, data_ori_rgb, data_ori_tif_l


def detection(data_augment_l, data_ori_rgb, tif_path, box_history_l, idx, process_type='Image'):
    bs = 32
    width, height = 640, 512
    pred_boxs = []
    for i in range(len(data_augment_l) // bs):
        im = data_augment_l[bs * i:bs * (i + 1)]
        im = np.array(im)
        im = torch.from_numpy(im).to(MODEL_DET.device)
        im = im.half() if MODEL_DET.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        im = torch.stack([im] * 3, dim=1)
        im = F.interpolate(im, size=(height, width), mode='bilinear', align_corners=False)
        pred = MODEL_DET(im, augment=ARGS.augment, visualize=False)
        pred = non_max_suppression(pred, ARGS.conf_thres, ARGS.iou_thres, ARGS.classes, ARGS.agnostic_nms,
                                   max_det=ARGS.max_det)
        pred_boxs.extend(pred)
    if len(data_augment_l) % bs != 0:
        i = len(data_augment_l) // bs
        im = data_augment_l[bs * i:]
        im = np.array(im)
        im = torch.from_numpy(im).to(MODEL_DET.device)
        im = im.half() if MODEL_DET.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        im = torch.stack([im] * 3, dim=1)
        im = F.interpolate(im, size=(height, width), mode='bilinear', align_corners=False)
        pred = MODEL_DET(im, augment=ARGS.augment, visualize=False)
        pred = non_max_suppression(pred, ARGS.conf_thres, ARGS.iou_thres, ARGS.classes, ARGS.agnostic_nms,
                                   max_det=ARGS.max_det)
        pred_boxs.extend(pred)

    pred_boxs = torch.cat(pred_boxs, dim=0)
    pred_boxs = pred_boxs.detach().cpu().numpy()
    pred_boxs = process_boxs(pred_boxs, width, height)
    if pred_boxs is None:
        return None, None, None, None, None, None, None, None, None
    color_l = [[255, 0, 0],
               [0, 255, 0],
               [0, 0, 255],
               [255, 255, 0],
               [255, 0, 255],
               ]
    data_ori_rgb = np.stack([data_ori_rgb] * 3, axis=-1)
    data_ori_rgb_copy = data_ori_rgb.copy()
    # pred_seg_info = []
    crop_img_l = []
    region_type_l = []
    box_l = []
    size_l = []
    device = MODEL_DET.device
    for box_id, box in enumerate(pred_boxs):
        confident, x_min, y_min, x_max, y_max, cls = box
        if confident > -0.5:
            box_history_l[int(cls)].append([float(x_min), float(y_min), float(x_max), float(y_max)])
            if len(box_history_l[int(cls)]) > 32:
                del (box_history_l[int(cls)][0])
            if len(box_history_l[int(cls)]) >= 3:
                data_all = np.array(box_history_l[int(cls)])
                data = data_all[:, :2].copy()
                data[:, 0:1] = (data_all[:, 0:1] + data_all[:, 2:3]) / 2.0
                data[:, 1:2] = (data_all[:, 1:2] + data_all[:, 3:4]) / 2.0

                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                mean = mean[np.newaxis]
                std = std[np.newaxis]

                z_scores = np.abs((data - mean) / (std + 1e-6))
                z_scores = np.mean(z_scores.reshape(-1, 2), axis=-1)
                threshold = np.sort(z_scores.ravel())[len(z_scores.ravel()) // 2]
                filtered_data = data_all[z_scores - threshold <= 1e-4]
                filtered_data = np.mean(filtered_data, axis=0)
                x_min, y_min, x_max, y_max = filtered_data[0], filtered_data[1], filtered_data[2], filtered_data[3]
                box = [confident, x_min, y_min, x_max, y_max, cls]

            cv2.rectangle(data_ori_rgb, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color_l[int(cls)], 2)

            region_type = torch.ones(1) * int(cls)
            region_type = region_type.unsqueeze(0).to(device)
            crop_img = data_ori_rgb_copy[int(y_min):int(y_max), int(x_min):int(x_max), :]
            crop_img = (torch.tensor(crop_img.transpose(2, 0, 1)).float().to(device) / 255.).unsqueeze(0)
            _, _, h, w = crop_img.shape
            crop_img = F.interpolate(crop_img, size=(128, 128), mode='bilinear', align_corners=False)
            crop_img_l.append(crop_img)
            region_type_l.append(region_type)
            box_l.append(box)
            size_l.append([h, w])

    output_box_save_path = None
    if process_type == 'Image' or ('Video' in process_type and idx == 0):
        output_box_save_path = os.path.join(tif_path, 'output_box.png')
        cv2.imwrite(output_box_save_path, data_ori_rgb)

    return crop_img_l, region_type_l, size_l, box_l, color_l, data_ori_rgb_copy, output_box_save_path, data_ori_rgb, box_history_l


def segmentation(crop_img_l, region_type_l, data_ori_rgb, data_ori_tif_l, size_l, box_l, color_l, tif_path, idx,
                 process_type='Image'):
    img = torch.cat(crop_img_l, dim=0)
    region_type = torch.stack(region_type_l, dim=0)
    time_count, output = MODEL_SEG.inference(img, region_type)
    segment_mask = np.zeros_like(data_ori_rgb)
    thresh_image_l = []
    data_ori_tif_l = np.stack(data_ori_tif_l, axis=-1)
    crop_tif_l = []
    for i in range(len(output)):
        h, w = size_l[i]
        confident, x_min, y_min, x_max, y_max, cls = box_l[i]

        out = output[i:i + 1]
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        thresh_image = (out > 0.5).float()
        thresh_image = thresh_image[0, 0].detach().cpu().numpy() * 255
        thresh_image_l.append(thresh_image.copy())
        crop_tif_l.append(data_ori_tif_l[int(y_min):int(y_max), int(x_min):int(x_max), :].copy())

        thresh_image = np.stack((thresh_image,) * 3, axis=-1)
        thresh_image_color = thresh_image * color_l[int(cls)]
        segment_mask[int(y_min):int(y_max), int(x_min):int(x_max)] += thresh_image_color

    visualized_image = cv2.addWeighted(data_ori_rgb, 0.9, segment_mask, 0.1, 0)
    output_seg_save_path = None
    if process_type == 'Image' or ('Video' in process_type and idx == 0):
        output_seg_save_path = os.path.join(tif_path, 'output_seg.png')
        cv2.imwrite(output_seg_save_path, visualized_image)

    return thresh_image_l, crop_tif_l, output_seg_save_path, visualized_image, data_ori_tif_l


def unmixing(box_l, thresh_image_l, crop_tif_l, data_ori_tif_l, tif_path, data_ori_rgb_copy, color_l, idx, A_history_l,
             process_type='Image'):
    ############################# obtain A ##############################
    A_l = []
    for i in range(len(box_l)):
        thresh_image = thresh_image_l[i] / 255.
        thresh_image = np.stack((thresh_image,) * 5, axis=-1)
        tif = crop_tif_l[i] * thresh_image
        height, width, channel = tif.shape
        border_width = int(0.2 * width)
        border_height = int(0.2 * height)

        if width <= 1 or width <= 1 or np.any(thresh_image[border_height:-border_height, border_width:-border_width].copy() == 0) == True:
            A = np.sum(tif, axis=(0, 1)) / np.sum(thresh_image, axis=(0, 1))
            A_l.append(A)
        else:
            tif = tif[border_height:-border_height, border_width:-border_width]
            thresh_image = thresh_image[border_height:-border_height, border_width:-border_width].copy()
            A = np.sum(tif, axis=(0, 1)) / np.sum(thresh_image, axis=(0, 1))
            A_l.append(A)

    # count = 0
    # bs = len(box_l)
    # box_count = 0
    # while count < 5 and box_count < bs:
    #     confident, x_min, y_min, x_max, y_max, cls = box_l[box_count]
    #     while count < int(cls):
    #         A_l.insert(count,np.array([1.0,2.0,3.0,4.0,5.0]))
    #         count = count + 1
    #     box_count = box_count + 1
    #     count = count + 1

    A_l = np.array(A_l)
    A_l = A_l.T
    h, w = A_l.shape
    # print('A_l_0', A_l)
    A_history_l.append(A_l)

    if len(A_history_l) > 32:
        del (A_history_l[0])

    if len(A_history_l) >= 3:
        data_all = np.array(A_history_l)
        data = np.array(A_history_l)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        mean = mean[np.newaxis]
        std = std[np.newaxis]
        z_scores = np.abs((data - mean) / (std + 1e-6))
        z_scores = np.mean(z_scores.reshape(-1, h * w), axis=-1)
        threshold = np.sort(z_scores.ravel())[len(z_scores.ravel()) // 2]
        filtered_data = data_all[z_scores - threshold <= 1e-4]
        filtered_data = np.mean(filtered_data, axis=0)
        A_l = filtered_data

    # print('A_l', A_l)

    output_txt_save_path = None
    if process_type == 'Image' or ('Video' in process_type and idx == 0):
        output_txt_save_path = os.path.join(tif_path, 'A.txt')
        np.savetxt(output_txt_save_path, A_l, fmt='%.4f', delimiter=' ')

    ########################### linear unmixing ##########################
    unmixing_part = ['tumor', 'intestine', 'colon', 'lymph', 'vessel']
    h, w, c = data_ori_tif_l.shape
    _, n = A_l.shape
    data_ori_tif_l = data_ori_tif_l.reshape(h * w, c)
    unminxing_results = (np.linalg.pinv(A_l) @ data_ori_tif_l.T).T
    unminxing_results = unminxing_results.reshape(h, w, n)
    unminxing_results[unminxing_results < 0] = 0
    directory, last_level = os.path.split(tif_path)
    unmixing_save_base = os.path.join(directory, last_level + '_results')
    os.makedirs(unmixing_save_base, exist_ok=True)
    unmixing_save_path_l = []
    unmixing_tif_save_path_l = []

    colored_img = data_ori_rgb_copy.copy()
    unminxing_results_l = []
    for i in range(unminxing_results.shape[2]):
        confident, x_min, y_min, x_max, y_max, cls = box_l[i]

        tif = unminxing_results[:, :, i]
        if process_type == 'Image' or ('Video' in process_type and idx == 0):
            unmixing_save_path = os.path.join(unmixing_save_base, 'unmixing_' + unmixing_part[int(cls)] + '.tif')
            unmixing_tif_save_path_l.append(unmixing_save_path)
            utility.save_tiff_imagej_compatible(unmixing_save_path, tif, 'YX')

        tif = (tif - tif.min()) / (tif.max() - tif.min()) * 255

        if process_type == 'Image' or ('Video' in process_type and idx == 0):
            unmixing_save_path = os.path.join(unmixing_save_base, 'unmixing_' + unmixing_part[int(cls)] + '.png')
            unmixing_save_path_l.append(unmixing_save_path)
            cv2.imwrite(unmixing_save_path, tif)

        normalized_tif = tif / 255.0
        for c in range(3):  # 对每个颜色通道进行叠加
            colored_img[:, :, c] = np.where(tif > 0,
                                            colored_img[:, :, c] * (1 - normalized_tif) + color_l[i][
                                                c] * normalized_tif,
                                            colored_img[:, :, c])
        tif = np.stack([tif] * 3, axis=-1)
        unminxing_results_l.append(tif)
    if process_type == 'Image' or ('Video' in process_type and idx == 0):
        unmixing_save_path = os.path.join(unmixing_save_base, 'unmixing_all.png')
        unmixing_save_path_l.insert(0, unmixing_save_path)
        cv2.imwrite(unmixing_save_path, colored_img)

    unminxing_results_l.insert(0, colored_img)

    return output_txt_save_path, unmixing_save_path_l, unmixing_tif_save_path_l, colored_img, unminxing_results_l, A_history_l


def create_video_from_images(images, video_path):
    frame = images[0]
    if len(frame.shape) == 3:
        height, width, layers = frame.shape
        isColor = True
    else:
        height, width = frame.shape
        isColor = False

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height), isColor)

    for image in images:
        video.write(image)

    video.release()
    return video_path


@torch.no_grad()
def run_model(tif_path, process_type):
    global MODEL_DET, ARGS, MODEL_SEG

    if MODEL_DET is None:
        gr.Error("MODEL_DET not load!")
        return [None, None, None, None, "MODEL_DET not load"]

    if MODEL_SEG is None:
        gr.Error("MODEL_SEG not load!")
        return [None, None, None, None, "MODEL_SEG not load"]

    files = os.listdir(tif_path)
    files.sort()
    count = 0
    tif_files = []
    tif_l = []
    for file in files:
        if '.tif' in file:
            tif_l.append(os.path.join(tif_path, file))
            count = count + 1
            if count % 5 == 0 and count != 0:
                tif_files.append(tif_l)
                tif_l = []

    box_history_l = []
    for i in range(5):
        box_history_l.append([])
    A_history_l = []

    # box_save_l = []
    # seg_save_l = []
    unmixing_l = []
    time_l = []
    for tif_idx in range(len(tif_files)):
        time_start = time.time()
        print('tif_idx', tif_idx)

        #####################################################################
        ############################# data augment ##########################
        #####################################################################
        data_augment_l, data_ori_rgb, data_ori_tif_l = data_generate_enhance(tif_files[tif_idx])

        #####################################################################
        ############################## detection ############################
        #####################################################################

        crop_img_l, region_type_l, size_l, box_l, color_l, data_ori_rgb_copy, output_box_save_path, data_ori_rgb, box_history_l = detection(
            data_augment_l, data_ori_rgb, tif_path, box_history_l, tif_idx, process_type=process_type)
        if crop_img_l is None:
            return [None, None, None, None, 'Cannot dectect five ROI areas.']
        # box_save_l.append(data_ori_rgb)
        #####################################################################
        ########################### segmentation ############################
        #####################################################################

        thresh_image_l, crop_tif_l, output_seg_save_path, visualized_image, data_ori_tif_l = segmentation(crop_img_l,
                                                                                                          region_type_l,
                                                                                                          data_ori_rgb,
                                                                                                          data_ori_tif_l,
                                                                                                          size_l, box_l,
                                                                                                          color_l,
                                                                                                          tif_path,
                                                                                                          tif_idx,
                                                                                                          process_type=process_type)
        # seg_save_l.append(visualized_image)
        #####################################################################
        ############################# unmixing ##############################
        #####################################################################

        output_txt_save_path, unmixing_save_path_l, unmixing_tif_save_path_l, colored_img, unminxing_results_l, A_history_l = unmixing(
            box_l, thresh_image_l, crop_tif_l, data_ori_tif_l, tif_path, data_ori_rgb_copy, color_l, tif_idx,
            A_history_l, process_type=process_type)
        unmixing_l.append(unminxing_results_l)

        time_end = time.time()
        time_l.append(time_end - time_start)

        if process_type == 'Image' or ('Video' in process_type and tif_idx == 0):
            output_AI_file_path = [output_box_save_path, output_seg_save_path, output_txt_save_path]
            output_AI_file_path.extend(unmixing_save_path_l)
            output_AI_file_path.extend(unmixing_tif_save_path_l)
            zip_save_path = os.path.join(tif_path, 'All_output_files.zip')
            with zipfile.ZipFile(zip_save_path, 'w') as zipf:
                for file in output_AI_file_path:
                    zipf.write(file, os.path.basename(file))
            output_AI_file_path.insert(0, zip_save_path)
            # output_AI_file_path = [zip_save_path]

            output_AI_visual_path = [output_box_save_path, output_seg_save_path]
            output_results_visual_path = []
            output_results_visual_path.extend(unmixing_save_path_l)
            # output_AI_path = [output_seg_save_path]

            output_txt_path = os.path.join(tif_path, 'readme.txt')
            readme_str = 'The video contains too many tif files, we offer the first five images for download.'
            with open(output_txt_path, "w", encoding="utf-8") as file:
                file.write(readme_str + "\n")
            output_AI_file_path.append(output_txt_path)

            if process_type == 'Image':
                return [output_AI_file_path, output_AI_visual_path, output_results_visual_path, None,
                        "Data: " + tif_path + ' has been unmixed.']

    frame_l = []
    for idx, imgs in enumerate(unmixing_l):
        frame_0 = np.concatenate(imgs[:3], axis=1)
        frame_1 = np.concatenate(imgs[3:], axis=1)
        frame = np.concatenate([frame_0, frame_1], axis=0)
        frame_l.append(frame.astype(np.uint8))
    video_path_all = create_video_from_images(frame_l, os.path.join(tif_path, 'unmixing_video.mp4'))

    with zipfile.ZipFile(zip_save_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(video_path_all, os.path.basename(video_path_all))
    output_AI_file_path.insert(1, video_path_all)

    for i in range(6):
        png_arrays_sub = []
        for idx, imgs in enumerate(unmixing_l):
            png_arrays_sub.append(imgs[i].astype(np.uint8))

        video_path = os.path.join(tif_path, 'unmixing_video_' + str(i).zfill(4) + '.mp4')
        video_path = create_video_from_images(png_arrays_sub, video_path)

        with zipfile.ZipFile(zip_save_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(video_path, os.path.basename(video_path))
        output_AI_file_path.insert(1, video_path)

    if len(time_l) > 20:
        time_l = time_l[20:]
    time_l = np.sum(time_l) / len(time_l)
    time_l = round(1 / time_l, 2)
    return [output_AI_file_path, output_AI_visual_path, output_results_visual_path, video_path_all,
            "Data: " + tif_path + ' has been unmixed.\nSpeed: ' + str(time_l) + 'fps.']


def get_tif_files(folder_path):
    tif_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tif')]
    tif_files.sort()
    return tif_files


def convert_tif_to_png(tif_files, process_type):
    png_files = []
    png_arrays = []
    for idx, tif_file in enumerate(tif_files):

        img = io.imread(tif_file)
        img_array = np.array(img)
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
        img_array = img_array.astype(np.uint8)
        png_arrays.append(img_array)
        png_file = tif_file.replace('.tif', '.png')
        if idx < 5:
            png_img = Image.fromarray(img_array)
            png_img.save(png_file, format='PNG')
        png_files.append(png_file)

    return png_files, png_arrays


def update_file_list(folder_path, custom_example):
    if 'Custom' in folder_path:
        return None, custom_example
    else:
        return None, ''

def convert_files(folder_path, process_type):
    tif_files = get_tif_files(folder_path)
    if process_type == 'Image':
        tif_files = tif_files[:5]
        png_files, png_arrays = convert_tif_to_png(tif_files, process_type)
        return png_files, png_files, None
    if 'Video' in process_type:
        png_files, png_arrays = convert_tif_to_png(tif_files, process_type)
        png_files = png_files[:len(png_files) // 5 * 5]
        png_arrays = png_arrays[:len(png_files) // 5 * 5]
        frame_l = []

        for i in range(len(png_files) // 5):
            png_arrays_sub_0 = np.concatenate([png_arrays[i * 5], png_arrays[i * 5 + 1], png_arrays[i * 5 + 2]], axis=1)
            ones_part = np.ones_like(png_arrays[i * 5]) * 255
            png_arrays_sub_1 = np.concatenate([png_arrays[i * 5 + 3], png_arrays[i * 5 + 4], ones_part], axis=1)
            png_arrays_sub = np.concatenate([png_arrays_sub_0, png_arrays_sub_1], axis=0)
            frame_l.append(png_arrays_sub)
        video_path_all = os.path.join(folder_path, 'video' + '.mp4')
        video_path_all = create_video_from_images(frame_l, video_path_all)


        files_name = png_files[:5]
        files_name.append(video_path_all)
        for i in range(5):
            png_arrays_sub = png_arrays[i::5]
            video_path = os.path.join(folder_path, 'video_' + str(i).zfill(4) + '.mp4')
            video_path = create_video_from_images(png_arrays_sub, video_path)
            files_name.append(video_path)
        output_txt_path = os.path.join(folder_path, 'readme.txt')
        readme_str = 'The video contains too many tif files, we offer the first five images for download.'
        with open(output_txt_path, "w", encoding="utf-8") as file:
            file.write(readme_str + "\n")
        files_name.append(output_txt_path)

        return files_name, png_files[:5], video_path_all


def device_state(device):
    global MODEL_DET, MODEL_SEG

    ARGS.device = device_select

    if device == 'CUDA':
        ARGS.device = '7'
    elif device == 'CPU':
        ARGS.device = 'cpu'

    if MODEL_DET is not None:
        device = select_device(ARGS.device)
        MODEL_DET = DetectMultiBackend(ARGS.weights, device=device, dnn=ARGS.dnn, data=ARGS.data, fp16=ARGS.half)
        MODEL_DET.eval()
        MODEL_SEG.device(device)

    return 'Model on %s' % (device)


def update_dropdowns(selection):
    if selection == "Image":
        return gr.update(visible=True, label="Input Image Viusalization"), gr.update(visible=False), gr.update(
            visible=True, label="Image Visualization Of Spectra Extraction"), gr.update(
            visible=False), selection, selection, 'Video needs to process many frames please wait.'
    elif "Video" in selection:
        return gr.update(visible=True, label="Input Image Viusalization (The First Five Tif Files)"), gr.update(
            visible=True), gr.update(visible=True,
                                     label="Image Visualization Of Spectra Extraction (The First Five Tif Files)"), gr.update(
            visible=True), selection, selection, 'Video needs to process many frames please wait.'



with gr.Blocks() as demo:
    gr.Markdown(
        "# Title: Real-time Deep Learning Spectral Imaging In vivo")
    # gr.Markdown(
    #     "# Title: AI-assisted high-capacity dynamic multiplexed imaging in vivo of monochromatic NIR-II-L fluorescence by excitation spectral resolving. ")
    gr.Markdown(
        "This demo allows you to run the models on your own images or the examples  from the paper. Please refer to the paper for more details.")

    gr.Markdown(
        "This is an online but non-real-time demo that processes all the spectral images before returning the results. We also provide a real-time unmixing version in the 'software' folder.")


    gr.Markdown("## Instructions")
    gr.Markdown(
        "1. We provide all test data here for display. You can also upload your zip file. The zip file contains the spectral images at each moment in chronological order, and the spectral image is in tif format.")
    gr.Markdown(
        "2. Select how you want the model to process the files. 'Image' denotes only processing the first five tif files, and 'Video' denotes processing all files. Video will take longer to process.")
    gr.Markdown(
        "3. Click 'Visualize Input' to see the input files. This will take a while to display the video.")
    gr.Markdown(
        "4. Click 'Load Model' to load the model. This may take a while.")
    gr.Markdown(
        "5. Click 'Extract Spectrum' to extract spectrum based on the inputs. This will take a while to display the video.")

    image_example_path = "exampledata/Unmixing/"
    image_custom_example_path = "exampledata/Unmixing_Custom/"
    os.makedirs(image_custom_example_path, exist_ok=True)
    subpaths = os.listdir(image_example_path)
    subpaths.sort()
    image_example_l = []
    for subpath in subpaths:
        if '_' not in subpath:
            image_example_l.append(os.path.join(image_example_path, subpath))
    custom_example_l = []

    gr.Markdown("## Upload or Use Examples")
    with gr.Row():
        image_upload = gr.File(label="Upload Custom Data Here", multiple=True, type="file", file_types=[".zip"],
                               interactive=True)


    with gr.Row():

        img_visual = gr.Gallery(label="Input Image Viusalization", show_label=True).style(grid=[3])

        video_visual = gr.Video(label="Input Video Viusalization", show_label=True, visible=False)

    with gr.Row():
        with gr.Column():
            input_message = gr.Textbox(label="Data Information", value=image_example_l[0])
            visualize_input = gr.Button("Visualize Input")
        with gr.Column():
            input_data_type = gr.Dropdown(label="Data Type", choices=['Image', 'Video (Video takes longer.)'],
                                          value="Image", interactive=True)
            reset_input = gr.Button("Reset Input")
        with gr.Column():
            custom_example = gr.Textbox(label="Custom Examples Information", value='', interactive=False)

            paper_examples = gr.Examples(
                label='Paper Examples List',
                examples=image_example_l,
                inputs=[input_message],
            )


    gr.Markdown("## Load and Run Model")

    with gr.Row().style():
        with gr.Column(scale=2):
            output_file = gr.File(label="Output File", interactive=False)
        with gr.Column(scale=1):
            ai_output = gr.Gallery(label="AI Detection and Segmentation Visualiztion", show_label=True).style(
                grid=[2], height="auto")

    with gr.Row():
        unmixing_output = gr.Gallery(label="Image Visualization Of Spectra Extraction", show_label=True).style(
            grid=[3], height="auto")

        unmixing_video_output = gr.Video(label="Video Visualization Of Spectra Extraction", show_label=True,
                                         visible=False).style(height="auto")

    with gr.Row():
        process_type = gr.Dropdown(label="Process Type", choices=['Image', 'Video (Video takes longer.)'],
                                   value="Image",
                                   interactive=True)
        device_select = gr.Dropdown(label="Device", choices=DEVICES, value="CUDA", interactive=True)
        load_progress = gr.Textbox(label="Model Information", value="Model not loaded")


    with gr.Row():
        output_message = gr.Textbox(label="Output Information", value="")
        load_btn = gr.Button("Load Model")
        run_btn = gr.Button("Extract Spectrum")

    input_message.change(fn=update_file_list, inputs=[input_message, custom_example],
                         outputs=[image_upload, custom_example])

    input_data_type.change(fn=update_dropdowns, inputs=input_data_type,
                           outputs=[img_visual, video_visual, unmixing_output, unmixing_video_output, input_data_type,
                                    process_type, output_message])
    process_type.change(fn=update_dropdowns, inputs=process_type,
                        outputs=[img_visual, video_visual, unmixing_output, unmixing_video_output, input_data_type,
                                 process_type, output_message])

    device_select.change(fn=device_state, inputs=device_select, outputs=output_message)
    visualize_input.click(fn=convert_files, inputs=[input_message, process_type],
                          outputs=[image_upload, img_visual, video_visual], queue=True)


    def reset_files():
        return image_example_l[0], 'Image'


    reset_input.click(fn=reset_files, inputs=None, outputs=[input_message, input_data_type], queue=True)

    load_btn.click(load_model, inputs=[process_type, device_select], outputs=load_progress, queue=True)
    run_btn.click(run_model, inputs=[input_message, process_type],
                  outputs=[output_file, ai_output, unmixing_output, unmixing_video_output, output_message], queue=True)


    def unzip_file(file):
        subfiles = os.listdir(image_custom_example_path)
        subfiles.sort()
        custom_dir = os.path.join(image_custom_example_path, str(len(subfiles)).zfill(4))
        os.makedirs(custom_dir, exist_ok=True)

        with zipfile.ZipFile(file.name, 'r') as zip_ref:
            zip_ref.extractall(custom_dir)

        old_files = os.listdir(custom_dir)
        old_files.sort()
        for idx, old_file in enumerate(old_files):
            src = os.path.join(custom_dir, old_file)
            tgt = os.path.join(custom_dir, str(idx).zfill(4) + '.tif')
            os.rename(src, tgt)

        return custom_dir


    def handle_upload(file, input_message, input_data_type, custom_example):
        if file is None:
            return input_message, custom_example, input_data_type
        if not file.name.endswith(".zip") or isinstance(file, list) == True:
            if 'Custom' not in input_message and isinstance(file, list) == False:
                return input_message, 'Please upload a zip file.', input_data_type
            elif 'Custom' not in input_message and isinstance(file, list) == True:
                return input_message, '', input_data_type
            else:
                return input_message, custom_example, input_data_type
        unzipped_path = unzip_file(file)

        return unzipped_path, 'Success upload the zip file. We save the file in: ' + unzipped_path + '.', input_data_type


    image_upload.change(fn=handle_upload, inputs=[image_upload, input_message, input_data_type, custom_example],
                        outputs=[input_message, custom_example, input_data_type])

    gr.Markdown("Copyright © 2024 School of Computer Science and Technology, Digital Media Lab, Fudan University. All rights reserved.", elem_id="footer", css_class="footer")

demo.queue().launch(server_name='0.0.0.0', server_port=7866)