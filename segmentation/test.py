import os
from collections import defaultdict

import cv2
from skimage import io
import numpy as np
from model.Segmentation import Model

import torch
from torch.nn import functional as F
from medpy.metric import binary

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

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

def process_files(folder_path):
    # 字典来存储每一组的数据，key为文件前缀，value为另一个字典（key为类别，value为最大置信度）
    group_confidences = defaultdict(lambda: defaultdict(lambda: (0, "")))
    box_category_l = []
    for i in range(5):
        box_category_l.append([])

    filenames = os.listdir(folder_path)
    filenames.sort()
    # for filename in filenames:
    #     print(filename)
    # exit(0)
    # 遍历文件夹中的所有文件
    prefix_curr = ''
    for filename in filenames:
        if filename.endswith(".txt"):
            # 获取文件前缀，假设格式如'0001_abc.txt'，分割以获取'0001'
            if '_' in filename:
                prefix = filename.split('_', 1)[0]
            else:
                prefix = filename.split('.', 1)[0]

            prefix_max = int(prefix)
            if prefix_curr == '':
                prefix_curr = prefix
            # if prefix_curr == '0010':
            #     exit(0)
            if prefix_curr != prefix:
                # print('prefix', prefix)
                # print('prefix_curr', prefix_curr)
                # print(group_confidences[prefix_curr])
                for category in range(5):
                    # print(category, 'category not in group_confidences[prefix_curr]', category not in group_confidences[prefix_curr])
                    if category in group_confidences[prefix_curr]:

                        confidence = group_confidences[prefix_curr][category][0]
                        line = group_confidences[prefix_curr][category][1]
                        # print(category)
                        # print('line', line)
                        x, y, w, h = line.split()[1:-1]
                        box_category_l[category].append([float(x), float(y), float(w), float(h)])


                    if len(box_category_l[category]) > 32:
                        del (box_category_l[category][0])

                    if len(box_category_l[category]) >= 3:
                        data_all = np.array(box_category_l[category])
                        data = data_all[:, :2]
                        # 计算每个维度的均值和标准差
                        mean = np.mean(data, axis=0)
                        std = np.std(data, axis=0)

                        # 计算 Z-score
                        z_scores = np.abs((data - mean) / (std + 1e-6))

                        z_scores = np.mean(z_scores.reshape(-1, 2), axis=-1)

                        threshold = np.sort(z_scores.ravel())[len(z_scores.ravel()) // 2]
                        # print('threshold', threshold)
                        # 筛选出非离群点
                        filtered_data = data_all[z_scores - threshold <= 1e-4]
                        filtered_data = np.mean(filtered_data, axis=0)

                        # # 设置阈值，选择合适的阈值以尽量去除约50%的离群点
                        # threshold = np.sort(z_scores.ravel())[len(z_scores.ravel()) // 2]
                        #
                        # # 筛选出非离群点
                        # filtered_data = data_all[( z_scores - threshold <= 1e-4 ).all(axis=1)]
                        # filtered_data = np.mean(filtered_data, axis=0)

                        update_line = [ str(category), str(filtered_data[0]), str(filtered_data[1]), str(filtered_data[2]), str(filtered_data[3]), str(confidence)]
                        update_line = ' '.join(update_line)
                        # print('update_line', update_line)
                        group_confidences[prefix_curr][category] = (0, update_line)

                    # print(prefix_curr,  group_confidences[prefix_curr][category])

                prefix_curr = prefix

                # for i in range(5):
                #     confidence, line =group_confidences[str(int(prefix)-1).zfill(4)][i]
                #     group_confidences[prefix][i] = (confidence, line)

            # 读取文件并处理每一行
            # print(prefix_curr)
            # exit(0)
            # if prefix_curr == '0000':

            with open(os.path.join(folder_path, filename), 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        category = int(parts[0])
                        confidence = float(parts[-1])
                        # 更新字典中的最大置信度
                        if confidence > group_confidences[prefix][category][0]:
                            group_confidences[prefix][category] = (confidence, line.strip())

            prefix_curr = prefix

    for category in range(5):
        # print(category, 'category not in group_confidences[prefix_curr]', category not in group_confidences[prefix_curr])
        if category in group_confidences[prefix_curr]:

            confidence = group_confidences[prefix_curr][category][0]
            line = group_confidences[prefix_curr][category][1]
            # print(category)
            # print('line', line)
            x, y, w, h = line.split()[1:-1]
            box_category_l[category].append([float(x), float(y), float(w), float(h)])


        if len(box_category_l[category]) > 32:
            del (box_category_l[category][0])

        if len(box_category_l[category]) >= 3:
            data_all = np.array(box_category_l[category])
            data = data_all[:, :2]
            # 计算每个维度的均值和标准差
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)

            # 计算 Z-score
            z_scores = np.abs((data - mean) / (std + 1e-6))

            z_scores = np.mean(z_scores.reshape(-1, 2), axis=-1)

            threshold = np.sort(z_scores.ravel())[len(z_scores.ravel()) // 2]
            # print('threshold', threshold)
            # 筛选出非离群点
            filtered_data = data_all[z_scores - threshold <= 1e-4]
            filtered_data = np.mean(filtered_data, axis=0)

            # # 设置阈值，选择合适的阈值以尽量去除约50%的离群点
            # threshold = np.sort(z_scores.ravel())[len(z_scores.ravel()) // 2]
            #
            # # 筛选出非离群点
            # filtered_data = data_all[( z_scores - threshold <= 1e-4 ).all(axis=1)]
            # filtered_data = np.mean(filtered_data, axis=0)

            update_line = [ str(category), str(filtered_data[0]), str(filtered_data[1]), str(filtered_data[2]), str(filtered_data[3]), str(confidence)]
            update_line = ' '.join(update_line)
            # print('update_line', update_line)
            group_confidences[prefix_curr][category] = (0, update_line)


    # for prefix, categories in group_confidences.items():
    #     category_l = []
    #     for category, (max_conf, line) in categories.items():
    #         category_l.append(category)
    #     # print(prefix, category_l)
    #     for j in range(5):
    #         if j not in category_l and prefix != '0001':
    #             group_confidences[prefix][j] = group_confidences[str(int(prefix)-1).zfill(4)][j]
    # 准备输出结果
    results = []
    for prefix, categories in group_confidences.items():
        # print(f"Prefix {prefix}:")
        for category, (max_conf, line) in categories.items():
            # print(f"  Category {category}: Max Confidence = {max_conf} | Line = {line}")
            results.append((prefix, category, max_conf, line))

    return results

gt_path = '../detection/data/unmixing/AllMask/'
gt_path_subpaths = os.listdir(gt_path)
gt_path_subpaths.sort()

label_path = '../detection/data/unmixing/labels/'

select_test = [5, 15, 18, 19, 25, 26, 29, 30, 31, 32]
select_valid = [20, 27, 24]
sample_id_l = [5, 15, 18, 19, 25, 26, 29, 30, 31, 32, 20, 27, 24]
dc_l_test = []
jc_l_test = []
dc_l_val = []
jc_l_val = []
os.makedirs('../metric', exist_ok=True)
f = open('../metric/results_iou.txt', 'w')
for sample_id in sample_id_l:

    dcs = []
    jcs = []

    sample_id = sample_id + 1
    f.write(str(sample_id).zfill(4) + '\n')
    # 文件夹路径
    folder_path = '../detection/runs/detect/unmixing_' +str(sample_id).zfill(2) + '/labels/'  # 更改为你的文件夹路径
    tif_path = '../detection/data/unmixing/fluorescence_time_data_tif/' +str(sample_id).zfill(4) + '/'
    tif_files_all = os.listdir(tif_path)
    tif_files_all.sort()
    save_path = '../metric/results_' +str(sample_id).zfill(2) + '/'
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model()
    model.load_model('train_log')
    model.eval()
    model.device(device)

    crop_d = 0.2

    # 调用函数
    max_confidences = process_files(folder_path)

    # 如果需要，可以进一步处理或输出max_confidences
    # print("All max confidences:", max_confidences)

    height, width = 512, 640

    # name = max_confidences[0][0]
    name_curr = max_confidences[0][0]
    save_line_l = []
    color_l = [ [0, 0, 255],
    [255, 0, 0],
    [255, 0, 255],
    [0, 255, 0],
    [255, 255, 0],
    ]
    class_l = ['tumor', 'intestine', 'colon', 'lymph', 'vessel']
    for max_confidence_idx, max_confidence in enumerate(max_confidences):
        # print(max_confidence)
        name = max_confidence[0]

        if name_curr != name:
            tif_files = tif_files_all[int(name_curr)*5:(int(name_curr)+1)*5]
            # tif_files = os.listdir(os.path.join(tif_path, str(int(name_curr)+1).zfill(4)))
            # tif_files.sort()

            tif_l = []
            for tif_file in tif_files:
                # input_path = os.path.join(tif_path, str(int(name_curr)+1).zfill(4), tif_file)
                input_path = os.path.join(tif_path, tif_file)
                # print(input_path)
                tif = io.imread(input_path)
                tif_array = np.array(tif)
                tif_l.append(tif_array)
            color_image = np.stack(tif_l, axis=2)
            color_image = np.max(color_image, axis=2)
            color_image = (color_image - np.min(color_image)) / (np.max(color_image) - np.min(color_image)) * 255

            color_image_mask = color_image.copy()

            color_image = np.stack((color_image,) * 3, axis=-1)
            color_image_copy = color_image.copy()
            color_image_copy_gt = color_image.copy()

            segment_mask = np.zeros_like(color_image)
            segment_mask_gt = np.zeros_like(color_image)
            for line_idx, line in enumerate(save_line_l):
                # print('class_l[int(c)]', class_l[int(c)])
                c, x_min, y_min, x_max, y_max = line
                cv2.rectangle(color_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color_l[int(c)], 2)

                ##
                # 打开文件并读取内容
                gt_label_xy = ''
                with open(os.path.join(label_path, str(sample_id-1).zfill(4) + '.txt'), 'r') as file:
                    # 逐行读取文件
                    for line in file:
                        # 去掉每行末尾的换行符
                        line = line.strip().replace('\n', '')
                        lines = line.split(' ')
                        if int(lines[0]) == int(c):
                            gt_label_xy = lines[1:]
                # print(gt_label_xy)
                gt_label_xy = [float(i) for i in gt_label_xy]
                x_gt, y_gt, w_gt, h_gt = gt_label_xy
                x_min_gt, y_min_gt, x_max_gt, y_max_gt =invert_convert((width, height), (float(x_gt), float(y_gt), float(w_gt), float(h_gt)))
                cv2.rectangle(color_image_copy_gt, (int(x_min_gt), int(y_min_gt)), (int(x_max_gt), int(y_max_gt)), color_l[int(c)], 2)

                cv2.imwrite(os.path.join(save_path, str(int(name_curr) + 1).zfill(4)+ '_' + class_l[int(c)] + '.png'), color_image_copy[int(y_min):int(y_max), int(x_min):int(x_max)])

                color_image_line = color_image_mask[int(y_min):int(y_max), int(x_min):int(x_max)].copy()
                # color_image_line = color_image_line.astype(np.uint8)
                height_binary, width_binary = color_image_line.shape


                color_image_line = np.stack((color_image_line,) * 3, axis=-1)
                # color_image_line = color_image_line.astype(np.uint8)
                color_image_line = (torch.tensor(color_image_line.transpose(2, 0, 1)).float().to(device) / 255.).unsqueeze(0)
                _,_,h,w = color_image_line.shape
                color_image_line = F.interpolate(color_image_line, size=(128, 128), mode='bilinear', align_corners=False)
                region_type = torch.ones(1) * int(c)
                region_type = region_type.unsqueeze(0).to(device)
                time_count, output = model.inference(color_image_line, region_type)
                output = F.interpolate(output, size=(h,w), mode='bilinear', align_corners=False)
                thresh_image = (output > 0.5).float()
                thresh_image = thresh_image[0, 0].detach().cpu().numpy() * 255
                cv2.imwrite(os.path.join(save_path, str(int(name_curr) + 1).zfill(4)+ '_' + class_l[int(c)] + '_mask.png'), thresh_image)

                crop_d = 0.2
                border_width = int(crop_d * width_binary)
                border_height = int(crop_d * height_binary)
                top, bottom, left, right = border_height, border_height, border_width, border_width
                thresh_image_shrink = thresh_image.copy()
                # thresh_image_shrink[:bottom] = 0
                # thresh_image_shrink[-top:] = 0
                # thresh_image_shrink[:, :left] = 0
                # thresh_image_shrink[:, -right:] = 0
                thresh_image_shrink[:left] = 0
                thresh_image_shrink[-right:] = 0
                thresh_image_shrink[:, :bottom] = 0
                thresh_image_shrink[:, -top:] = 0
                cv2.imwrite(
                    os.path.join(save_path, str(int(name_curr) + 1).zfill(4) + '_' + class_l[int(c)] + '_mask_shrink.png'),
                    thresh_image_shrink)

                thresh_image_metric = thresh_image.copy()
                thresh_image = np.stack((thresh_image,) * 3, axis=-1)
                thresh_image_color = thresh_image * color_l[int(c)]
                segment_mask[int(y_min):int(y_max), int(x_min):int(x_max)] += thresh_image_color

                segment_mask_metric = np.zeros_like(color_image)[:,:,0]
                segment_mask_metric[int(y_min):int(y_max), int(x_min):int(x_max)] = thresh_image_metric



                cv2.imwrite(
                    os.path.join(save_path, str(int(name_curr) + 1).zfill(4) + '_' + class_l[int(c)] + '_mask_metric.png'),
                    segment_mask_metric)

                gt_metric_image = os.path.join(gt_path, gt_path_subpaths[int(c)], str(sample_id-1).zfill(4)+'.png')
                # print('gt_metric_image', gt_metric_image)
                gt_metric_image = cv2.imread(gt_metric_image)
                gt_metric_image_color = (gt_metric_image.copy()) * color_l[int(c)]
                segment_mask_gt += gt_metric_image_color

                cv2.imwrite(
                    os.path.join(save_path, str(int(name_curr) + 1).zfill(4) + '_' + class_l[int(c)] + '_mask_gt_metric.png'),
                    gt_metric_image)

                segment_mask_metric = (segment_mask_metric//255. + 0.5).astype(np.int32).astype(bool)
                gt_metric_image = (gt_metric_image//255. + 0.5).astype(np.int32).astype(bool)

                if np.sum(gt_metric_image) == 0:
                    dc = -1
                    jc = -1
                else:
                    dc = binary.dc(segment_mask_metric, gt_metric_image[:,:,0])
                    jc = binary.jc(segment_mask_metric, gt_metric_image[:,:,0])
                dcs.append(dc)
                jcs.append(jc)
                # exit(0)

                # border_width = int(crop_d * width_binary)
                # border_height = int(crop_d * height_binary)
                #
                # image = color_image_line[border_height:-border_height, border_width:-border_width].copy()
                # # 应用Otsu的二值化
                # _, thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # # 显示结果
                # top, bottom, left, right = border_height, border_height, border_width, border_width
                # border_type = cv2.BORDER_CONSTANT
                # thresh_image = cv2.copyMakeBorder(thresh_image, top, bottom, left, right, border_type, value=0)
                #
                # cv2.imwrite(os.path.join(save_path, str(int(name_curr) + 1).zfill(4)+ '_' + class_l[int(c)] + '_mask.png'), thresh_image)


                for tif_idx, tif_save in enumerate(tif_l):
                    tif_save = tif_save[int(y_min):int(y_max), int(x_min):int(x_max)]
                    io.imsave(os.path.join(save_path, str(int(name_curr)+1).zfill(4) + '_' + class_l[int(c)] + '_' + str(tif_idx).zfill(2) + '.tif'), tif_save)


            cv2.imwrite(os.path.join(save_path, str(int(name_curr)+1).zfill(4) + '.png'), color_image)
            visualized_image = cv2.addWeighted(color_image, 0.9, segment_mask, 0.1, 0)
            cv2.imwrite(os.path.join(save_path, str(int(name_curr) + 1).zfill(4) + '_segment.png'), visualized_image)

            visualized_image = cv2.addWeighted(color_image_copy_gt, 0.9, segment_mask_gt, 0.1, 0)
            cv2.imwrite(os.path.join(save_path, str(int(name_curr) + 1).zfill(4) + '_segment_gt.png'), visualized_image)

            name_curr = max_confidence[0]
            save_line_l = []

        x, y, w, h = max_confidence[3].split()[1:-1]
        x_min, y_min, x_max, y_max = invert_convert((width, height), (float(x), float(y), float(w), float(h)))
        c = max_confidence[1]
        save_line_l.append([c, x_min, y_min, x_max, y_max])


    tif_files = tif_files_all[int(name_curr)*5:(int(name_curr)+1)*5]
    # tif_files = os.listdir(os.path.join(tif_path, str(int(name_curr)+1).zfill(4)))
    # tif_files.sort()

    tif_l = []
    for tif_file in tif_files:
        # input_path = os.path.join(tif_path, str(int(name_curr)+1).zfill(4), tif_file)
        input_path = os.path.join(tif_path, tif_file)
        tif = io.imread(input_path)
        tif_array = np.array(tif)
        tif_l.append(tif_array)
    color_image = np.stack(tif_l, axis=2)
    color_image = np.max(color_image, axis=2)
    color_image = (color_image - np.min(color_image)) / (np.max(color_image) - np.min(color_image)) * 255

    color_image_mask = color_image.copy()

    color_image = np.stack((color_image,) * 3, axis=-1)
    color_image_copy = color_image.copy()
    segment_mask = np.zeros_like(color_image)
    for line_idx, line in enumerate(save_line_l):
        c, x_min, y_min, x_max, y_max = line
        cv2.rectangle(color_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color_l[int(c)], 2)
        cv2.imwrite(os.path.join(save_path, str(int(name_curr) + 1).zfill(4)+ '_' + class_l[int(c)] + '.png'), color_image_copy[int(y_min):int(y_max), int(x_min):int(x_max)])

        color_image_line = color_image_mask[int(y_min):int(y_max), int(x_min):int(x_max)].copy()
        # color_image_line = color_image_line.astype(np.uint8)
        height_binary, width_binary = color_image_line.shape

        color_image_line = np.stack((color_image_line,) * 3, axis=-1)
        # color_image_line = color_image_line.astype(np.uint8)
        color_image_line = (torch.tensor(color_image_line.transpose(2, 0, 1)).float().to(device) / 255.).unsqueeze(0)
        _, _, h, w = color_image_line.shape
        color_image_line = F.interpolate(color_image_line, size=(128, 128), mode='bilinear', align_corners=False)
        region_type = torch.ones(1) * int(c)
        region_type = region_type.unsqueeze(0).to(device)
        time_count, output = model.inference(color_image_line, region_type)
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
        thresh_image = (output > 0.5).float()
        thresh_image = thresh_image[0, 0].detach().cpu().numpy() * 255
        cv2.imwrite(os.path.join(save_path, str(int(name_curr) + 1).zfill(4) + '_' + class_l[int(c)] + '_mask.png'),
                    thresh_image)

        crop_d = 0.2
        border_width = int(crop_d * width_binary)
        border_height = int(crop_d * height_binary)
        top, bottom, left, right = border_height, border_height, border_width, border_width
        thresh_image_shrink = thresh_image.copy()
        # thresh_image_shrink[:bottom] = 0
        # thresh_image_shrink[-top:] = 0
        # thresh_image_shrink[:, :left] = 0
        # thresh_image_shrink[:, -right:] = 0
        thresh_image_shrink[:left] = 0
        thresh_image_shrink[-right:] = 0
        thresh_image_shrink[:, :bottom] = 0
        thresh_image_shrink[:, -top:] = 0
        cv2.imwrite(
            os.path.join(save_path, str(int(name_curr) + 1).zfill(4) + '_' + class_l[int(c)] + '_mask_shrink.png'),
            thresh_image_shrink)

        thresh_image_metric = thresh_image.copy()
        thresh_image = np.stack((thresh_image,) * 3, axis=-1)
        thresh_image_color = thresh_image * color_l[int(c)]
        segment_mask[int(y_min):int(y_max), int(x_min):int(x_max)] += thresh_image_color

        # border_width = int(crop_d * width_binary)
        # border_height = int(crop_d * height_binary)
        #
        # image = color_image_line[border_height:-border_height, border_width:-border_width].copy()
        # # 应用Otsu的二值化
        # _, thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # # 显示结果
        # top, bottom, left, right = border_height, border_height, border_width, border_width
        # border_type = cv2.BORDER_CONSTANT
        # thresh_image = cv2.copyMakeBorder(thresh_image, top, bottom, left, right, border_type, value=0)
        #
        # cv2.imwrite(os.path.join(save_path, str(int(name_curr) + 1).zfill(4) + '_' + class_l[int(c)] + '_mask.png'),
        #             thresh_image)

        segment_mask_metric = np.zeros_like(color_image)[:, :, 0]
        segment_mask_metric[int(y_min):int(y_max), int(x_min):int(x_max)] = thresh_image_metric
        cv2.imwrite(
            os.path.join(save_path, str(int(name_curr) + 1).zfill(4) + '_' + class_l[int(c)] + '_mask_metric.png'),
            segment_mask_metric)

        gt_metric_image = os.path.join(gt_path, gt_path_subpaths[int(c)], str(sample_id - 1).zfill(4) + '.png')
        gt_metric_image = cv2.imread(gt_metric_image)
        cv2.imwrite(
            os.path.join(save_path, str(int(name_curr) + 1).zfill(4) + '_' + class_l[int(c)] + '_mask_gt_metric.png'),
            gt_metric_image)

        segment_mask_metric = (segment_mask_metric // 255. + 0.5).astype(np.int32).astype(bool)
        gt_metric_image = (gt_metric_image // 255. + 0.5).astype(np.int32).astype(bool)

        if np.sum(gt_metric_image) == 0:
            dc = -1
            jc = -1
        else:
            dc = binary.dc(segment_mask_metric, gt_metric_image[:, :, 0])
            jc = binary.jc(segment_mask_metric, gt_metric_image[:, :, 0])
        dcs.append(dc)
        jcs.append(jc)


        for tif_idx, tif in enumerate(tif_l):
            tif = tif[int(y_min):int(y_max), int(x_min):int(x_max)]
            io.imsave(os.path.join(save_path,
                                   str(int(name_curr) + 1).zfill(4) + '_' + class_l[int(c)] + '_' + str(tif_idx).zfill(
                                       2) + '.tif'), tif)
            tif_shrink = tif[border_height:-border_height, border_width:-border_width].copy()


    cv2.imwrite(os.path.join(save_path, str(int(name_curr)+1).zfill(4) + '.png'), color_image)
    visualized_image = cv2.addWeighted(color_image, 0.9, segment_mask, 0.1, 0)
    cv2.imwrite(os.path.join(save_path, str(int(name_curr) + 1).zfill(4) + '_segment.png'), visualized_image)

    print('*' * 10)
    print('dc', sum(dcs) / len(dcs))
    print('jc', sum(jcs) / len(jcs))
    f.write(str(sum(jcs) / len(jcs)) + '\n\n')
    if sample_id in select_test:
        dc_l_test.append(sum(dcs) / len(dcs))
        jc_l_test.append(sum(jcs) / len(jcs))
    if sample_id in select_valid:
        dc_l_val.append(sum(dcs) / len(dcs))
        jc_l_val.append(sum(jcs) / len(jcs))

print()
print('average')
print('All average dc test', sum(dc_l_test)/len(dc_l_test))
print('All average jc test', sum(jc_l_test)/len(jc_l_test))

print('All average dc val', sum(dc_l_val)/len(dc_l_val))
print('All average jc val', sum(jc_l_val)/len(jc_l_val))

f.write('Average test\n')
f.write(str(sum(jc_l_test)/len(jc_l_test)) + '\n\n')

f.write('Average val\n')
f.write(str(sum(jc_l_val)/len(jc_l_val)) + '\n\n')