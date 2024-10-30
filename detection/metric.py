import os
import cv2
import numpy as np
import math

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


def calculate_iou(pred_box, gt_box):
    pred_x_min, pred_y_min, pred_x_max, pred_y_max = pred_box
    gt_y_max, gt_y_min, gt_x_max, gt_y_max =gt_box
    # gt_x_min, gt_y_min = gt_box[:2]
    # gt_x_max = gt_x_min + gt_box[2]
    # gt_y_max = gt_y_min + gt_box[3]
    # print('pred_x_min, pred_y_min, pred_x_max, pred_y_max', pred_x_min, pred_y_min, pred_x_max, pred_y_max)
    # print('gt_y_max, gt_y_min, gt_x_max, gt_y_max', gt_y_max, gt_y_min, gt_x_max, gt_y_max)
    # 计算交集
    intersect_x_min = max(pred_x_min, gt_x_min)
    intersect_y_min = max(pred_y_min, gt_y_min)
    intersect_x_max = min(pred_x_max, gt_x_max)
    intersect_y_max = min(pred_y_max, gt_y_max)

    if intersect_x_min < intersect_x_max and intersect_y_min < intersect_y_max:
        intersect_area = (intersect_x_max - intersect_x_min) * (intersect_y_max - intersect_y_min)
    else:
        intersect_area = 0

    # 计算并集
    pred_area = (pred_x_max - pred_x_min) * (pred_y_max - pred_y_min)
    gt_area = (gt_x_max - gt_x_min) * (gt_y_max - gt_y_min)
    union_area = pred_area + gt_area - intersect_area

    # 计算IoU
    iou = intersect_area / union_area

    return iou


def calculate_ap(recall, precision):
    recall = np.concatenate(([0.], recall, [1.]))
    precision = np.concatenate(([0.], precision, [0.]))

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap


def calculate_map(all_predictions, all_gt_boxes, iou_threshold=0.5):
    # all_predictions: List of lists of tuples (confidence_score, pred_box) for each image
    # all_gt_boxes: List of lists of ground truth boxes for each image

    all_ap = []
    for predictions, gt_boxes in zip(all_predictions, all_gt_boxes):
        # print('predictions', predictions)
        # print('gt_boxes', gt_boxes)
        predictions.sort(key=lambda x: x[0], reverse=True)  # 按置信度降序排序

        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        detected = []

        for i, (confidence, pred_box) in enumerate(predictions):
            best_iou = 0
            best_gt_idx = -1
            for j, gt_box in enumerate(gt_boxes):
                # print('pred_box', pred_box)
                # print('gt_box', gt_box)
                iou = calculate_iou(pred_box, gt_box)
                # print('iou', iou)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx not in detected:
                tp[i] = 1  # True Positive
                detected.append(best_gt_idx)
            else:
                fp[i] = 1  # False Positive

        # 累积 TP 和 FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        # 计算 Precision 和 Recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / len(gt_boxes)
        # print('precision', precision)
        # print('recall', recall)
        # 计算 AP
        ap = calculate_ap(recall, precision)
        all_ap.append(ap)

    # 计算 mAP
    map_score = np.mean(all_ap)
    return map_score


# Path to the directory containing the .txt files
directory = './runs/detect/'
directory_image = './data/unmixing/fluorescence_time_data/'
gt_path = './data/unmixing/labels/'
# Dictionary to store the highest confidence and corresponding coordinates for each class
# select_test = [5, 15, 17, 18, 19, 25, 26]
select_test = [5, 15, 18, 19, 25, 26, 29, 30, 31, 32]
select_valid = [20, 27, 24]
select_all = [5, 15, 18, 19, 25, 26, 29, 30, 31, 32, 20, 27, 24]
# select_all  = [5, 15]

height, width = 512, 640

iou_val = []
iou_test = []
iou_all_val = []
iou_all_test = []
# Iterate over each file in the directory
for idx, select_id in enumerate(select_all):
    iou_tmp = []
    print('*' * 20)
    print(select_id)

    select_id_plus1 = str(select_id+1).zfill(4)
    select_id_plus1_2 = str(select_id+1).zfill(2)
    select_id = str(select_id).zfill(4)

    directory_image_path = os.path.join(directory_image, select_id_plus1)
    image_names = os.listdir(directory_image_path)
    image_names.sort()
    # print('directory_image_path', directory_image_path)
    # print('image_names[-1]', image_names[-1])
    frame_num = int(image_names[-1][:image_names[-1].find('_')])
    # print('frame_num', frame_num)


    for frame_idx in range(frame_num+1):
        highest_confidence_boxes = {0: None, 1: None, 2: None, 3: None, 4: None}
        frame_idx = str(frame_idx).zfill(4)

        directory_path = os.path.join(directory, 'unmixing_' +  select_id_plus1_2, 'labels')
        # print('len(os.listdir(directory_path))', len(os.listdir(directory_path)))
        filenames = os.listdir(directory_path)
        filenames.sort()
        for filename in filenames:
            # print('filename', filename, 'frame_idx', frame_idx)
            if filename.startswith(frame_idx) and filename.endswith('.txt'):
                filepath = os.path.join(directory_path, filename)
                # print('filepath', filepath)
                with open(filepath, 'r') as file:
                    for line in file:
                        data = line.replace('\n', '').strip().split()
                        class_id = int(data[0])
                        coords = list(map(float, data[1:5]))
                        confidence = float(data[5])

                        # Check if this is the highest confidence seen so far for this class
                        if (highest_confidence_boxes[class_id] is None or
                                confidence > highest_confidence_boxes[class_id][1]):
                            highest_confidence_boxes[class_id] = (coords, confidence)
        # print('highest_confidence_boxes', highest_confidence_boxes)
        # get gt
        all_predictions = []
        all_gt_boxes = []
        for ii in range(5):
            all_predictions.append([])
            all_gt_boxes.append([])

        gt_boxs = {0: None, 1: None, 2: None, 3: None, 4: None}
        with open(os.path.join(gt_path, select_id+'.txt'), 'r') as file:
            for line in file:
                data = line.replace('\n', '').strip().split()
                class_id_gt = int(data[0])
                coords_gt = list(map(float, data[1:5]))
                gt_boxs[class_id_gt] = coords_gt


        # img = cv2.imread(os.path.join(image_path, select_id+'.png'))
        # img_gt = img.copy()
        # Print the results
        iou_l = []
        for class_id, box in highest_confidence_boxes.items():
            if box is not None:
                coords, confidence = box
                x, y, w, h = coords
                x_min, y_min, x_max, y_max = invert_convert((width, height), (float(x), float(y), float(w), float(h)))
                # print(f'Class {class_id}: Original coordinates: ({x_min:.2f}, {y_min:.2f}), ({x_max:.2f}, {y_max:.2f}, Confidence: {confidence}')
                # print(f'Class {class_id}: Coordinates: {coords}, Confidence: {confidence}')
                # print(f'  Original coordinates: ({x_min:.2f}, {y_min:.2f}), ({x_max:.2f}, {y_max:.2f})')
                # cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color_l[class_id], 2)

                gt_coords = gt_boxs[class_id]
                gt_x, gt_y, gt_w, gt_h = gt_coords
                gt_x_min, gt_y_min, gt_x_max, gt_y_max = invert_convert((width, height), (float(gt_x), float(gt_y), float(gt_w), float(gt_h)))
                # cv2.rectangle(img_gt, (int(gt_x_min), int(gt_y_min)), (int(gt_x_max), int(gt_y_max)), color_l[class_id], 2)
                # print(f'Class {class_id}: gt coordinates: ({gt_x_min:.2f}, {gt_y_min:.2f}), ({gt_x_max:.2f}, {gt_y_max:.2f}')
                # print()

                iou = calculate_iou([x_min, y_min, x_max, y_max], [gt_x_min, gt_y_min, gt_x_max, gt_y_max])
                # print('iou', iou)
                iou_l.append(iou)
                if select_all[idx] in select_test:
                    iou_test.append(iou)
                else:
                    iou_val.append(iou)
                iou_tmp.append(iou)
                all_gt_boxes[class_id].append([gt_x_min, gt_y_min, gt_x_max, gt_y_max])
                all_predictions[class_id].append((confidence, [x_min, y_min, x_max, y_max]))

            else:
                iou=0
                iou_l.append(iou)
                if select_all[idx] in select_test:
                    iou_test.append(iou)
                else:
                    iou_val.append(iou)
                iou_tmp.append(iou)
                # print(f'Class {class_id}: No detections found.')

        # print('mean iou:', sum(iou_l)/len(iou_l))

        # if len(iou_test) != 0:
        #     print('average iou test: ', sum(iou_test)/len(iou_test))
        # if len(iou_val) != 0:
        #     print('average iou val: ', sum(iou_val)/len(iou_val))


    if select_all[idx] in select_test:
        iou_all_test.append(sum(iou_tmp)/len(iou_tmp))
        print('test: ', select_id, sum(iou_tmp)/len(iou_tmp))
    else:
        iou_all_val.append(sum(iou_tmp)/len(iou_tmp))
        print('val: ', select_id, sum(iou_tmp) / len(iou_tmp))

# print('average iou test: ', sum(iou_test)/len(iou_test))
# print('average iou val: ', sum(iou_val)/len(iou_val))

print('average iou test: ', sum(iou_all_test)/len(iou_all_test))
print('average iou val: ', sum(iou_all_val)/len(iou_all_val))