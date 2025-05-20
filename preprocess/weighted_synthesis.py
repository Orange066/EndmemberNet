import os
import cv2
import numpy as np
from PIL import Image
import time
from skimage import io

def ori_augmentation(input_folder_0, input_folder_1, output_folder):
    """
    递归处理文件夹下的所有 TIFF 文件，并将它们转换为 PNG 文件

    Args:
    - input_folder: 输入文件夹路径，包含 TIFF 文件和子文件夹
    - output_folder: 输出文件夹路径，保存转换后的 PNG 文件
    """

    subfiles = os.listdir(input_folder_0)
    subfiles.sort()
    count = 0
    for subfile_id, subfile in enumerate(subfiles):
        if 'mask' not in subfile:
            files = os.listdir(os.path.join(input_folder_0, subfile))
            files.sort()

            img_l = []
            for file_id, file in enumerate(files[:5]):

                input_path = os.path.join(input_folder_0, subfile, file)

                img = io.imread(input_path)
                img_array = np.array(img)

                img_l.append(img_array)

            img_array = np.stack(img_l, axis=2)
            img_array = np.max(img_array, axis=2)

            os.makedirs(os.path.join(output_folder, str(count).zfill(4)), exist_ok=True)

            output_folder_tmp = os.path.join(output_folder, str(count).zfill(4), str(count).zfill(4) + '.png')
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
            cv2.imwrite(output_folder_tmp, img_array)

            # eqh
            img_array = img_array.astype(np.uint8)
            image = img_array
            image_copy = img_array.copy()
            num_min = int(np.min(image)) - 1
            num_max = int(np.max(image)) + 1
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
            cv2.imwrite(output_folder_tmp.replace('.png', '_eqh.png'), result_image)

            count += 1

    # subfiles = os.listdir(input_folder_1)
    # subfiles.sort()
    # for subfile_id, subfile in enumerate(subfiles):
    #     if 'mask' not in subfile:
    #         files = os.listdir(os.path.join(input_folder_1, subfile))
    #         files.sort()

    #         img_l = []
    #         for file_id, file in enumerate(files[:5]):

    #             input_path = os.path.join(input_folder_1, subfile, file)

    #             img = io.imread(input_path)
    #             img_array = np.array(img)

    #             img_l.append(img_array)

    #         img_array = np.stack(img_l, axis=2)
    #         img_array = np.max(img_array, axis=2)

    #         os.makedirs(os.path.join(output_folder, str(count).zfill(4)), exist_ok=True)
    #         output_folder_tmp = os.path.join(output_folder, str(count).zfill(4), str(count).zfill(4) + '.png')
    #         img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
    #         cv2.imwrite(output_folder_tmp, img_array)


    #         # eqh
    #         img_array = img_array.astype(np.uint8)
    #         image = img_array
    #         image_copy = img_array.copy()
    #         num_min = int(np.min(image)) - 1
    #         num_max = int(np.max(image)) + 1
    #         hist, bin_edges = np.histogram(image, bins=20, range=(num_min, num_max))

    #         hist_max = np.max(hist)
    #         delete_idx = 0
    #         for h_idx, h in enumerate(hist):
    #             if h > hist_max // 2:
    #                 delete_idx = h_idx

    #         delete_hist = bin_edges[delete_idx + 1]
    #         image_copy[image_copy < delete_hist] = 0
    #         image_copy[image_copy >= delete_hist] = 1

    #         # 对于灰度图像，直接应用均衡化
    #         image = image * image_copy
    #         result_image = cv2.equalizeHist(image)
    #         cv2.imwrite(output_folder_tmp.replace('.png', '_eqh.png'), result_image)

    #         count += 1


def weighted_synthesis(input_folder_0, input_folder_1, output_folder):
    """
    递归处理文件夹下的所有 TIFF 文件，并将它们转换为 PNG 文件

    Args:
    - input_folder: 输入文件夹路径，包含 TIFF 文件和子文件夹
    - output_folder: 输出文件夹路径，保存转换后的 PNG 文件
    """

    subfiles = os.listdir(input_folder_0)
    subfiles.sort()
    count = 0
    for subfile in subfiles:
        if 'mask' not in subfile:
            files = os.listdir(os.path.join(input_folder_0, subfile))
            files.sort()

            img_l = []
            for file_id, file in enumerate(files[:5]):

                input_path = os.path.join(input_folder_0, subfile, file)

                img = io.imread(input_path)
                img_array = np.array(img)
                img_l.append(img_array)

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

            for loc_0 in range(5):
                select_l_tmp = [1, 1, 1, 1, 1]
                select_l_tmp[loc_0] = 0
                select_l.append(select_l_tmp)

            for loc_0 in range(5):
                select_l_tmp = [1, 1, 1, 1, 1]
                select_l.append(select_l_tmp)

            # for select in select_l:
            #     print(select)
            # exit(0)

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

                os.makedirs(os.path.join(output_folder, str(count).zfill(4)), exist_ok=True)

                output_folder_tmp = os.path.join(output_folder, str(count).zfill(4), str(count).zfill(4) +'_' + str(j).zfill(4) + '.png')
                img_array = (weighted_sum - weighted_sum.min()) / (weighted_sum.max() - weighted_sum.min()) * 255
                cv2.imwrite(output_folder_tmp, img_array)


                image = img_array.astype(np.uint8)
                image_copy = image.copy()
                num_min = int(np.min(image)) - 1
                num_max = int(np.max(image)) + 1
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
                # print('output_folder_tmp_0:', output_folder_tmp)
                cv2.imwrite(output_folder_tmp.replace('.png', '_eqh.png'), result_image)

            count += 1

    # subfiles = os.listdir(input_folder_1)
    # subfiles.sort()
    # for subfile in subfiles:
    #     if 'mask' not in subfile:
    #         files = os.listdir(os.path.join(input_folder_1, subfile))
    #         files.sort()

    #         img_l = []
    #         for file_id, file in enumerate(files[:5]):
    #             input_path = os.path.join(input_folder_1, subfile, file)

    #             img = io.imread(input_path)
    #             img_array = np.array(img)
    #             img_l.append(img_array)

    #         select_l = []
    #         for loc_0 in range(5):
    #             for loc_1 in range(loc_0 + 1, 5):
    #                 select_l_tmp = [0, 0, 0, 0, 0]
    #                 select_l_tmp[loc_1] = 1
    #                 select_l_tmp[loc_0] = 1
    #                 select_l.append(select_l_tmp)

    #         for loc_0 in range(5):
    #             select_l_tmp = [0, 0, 0, 0, 0]
    #             select_l_tmp[loc_0] = 1
    #             select_l.append(select_l_tmp)

    #         for loc_0 in range(5):
    #             select_l_tmp = [1, 1, 1, 1, 1]
    #             select_l_tmp[loc_0] = 0
    #             select_l.append(select_l_tmp)

    #         for loc_0 in range(5):
    #             select_l_tmp = [1, 1, 1, 1, 1]
    #             select_l.append(select_l_tmp)

    #         # for select in select_l:
    #         #     print(select)
    #         # exit(0)

    #         for j in range(len(select_l)):
    #             # 生成五个随机权重
    #             weights = np.random.rand(5)
    #             weights[0] = weights[0] * select_l[j][0]
    #             weights[1] = weights[1] * select_l[j][1]
    #             weights[2] = weights[2] * select_l[j][2]
    #             weights[3] = weights[3] * select_l[j][3]
    #             weights[4] = weights[4] * select_l[j][4]

    #             # 使用权重对数组进行加权求和
    #             weighted_sum = np.zeros_like(img_l[0])
    #             for arr, weight in zip(img_l, weights):
    #                 weighted_sum += arr * weight

    #             os.makedirs(os.path.join(output_folder, str(count).zfill(4)), exist_ok=True)

    #             output_folder_tmp = os.path.join(output_folder, str(count).zfill(4),
    #                                              str(count).zfill(4) + '_' + str(j).zfill(4) + '.png')
    #             img_array = (weighted_sum - weighted_sum.min()) / (
    #                         weighted_sum.max() - weighted_sum.min()) * 255
    #             cv2.imwrite(output_folder_tmp, img_array)

    #             image = img_array.astype(np.uint8)
    #             image_copy = image.copy()
    #             num_min = int(np.min(image)) - 1
    #             num_max = int(np.max(image)) + 1
    #             hist, bin_edges = np.histogram(image, bins=20, range=(num_min, num_max))

    #             hist_max = np.max(hist)
    #             delete_idx = 0
    #             for h_idx, h in enumerate(hist):
    #                 if h > hist_max // 2:
    #                     delete_idx = h_idx

    #             delete_hist = bin_edges[delete_idx + 1]
    #             image_copy[image_copy < delete_hist] = 0
    #             image_copy[image_copy >= delete_hist] = 1

    #             # 对于灰度图像，直接应用均衡化
    #             image = image * image_copy
    #             result_image = cv2.equalizeHist(image)
    #             cv2.imwrite(output_folder_tmp.replace('.png', '_eqh.png'), result_image)
    #             # print('output_folder_tmp_1:', output_folder_tmp)
    #         count += 1


def visualize(input_folder_0, input_folder_1, output_folder):
    """
    递归处理文件夹下的所有 TIFF 文件，并将它们转换为 PNG 文件

    Args:
    - input_folder: 输入文件夹路径，包含 TIFF 文件和子文件夹
    - output_folder: 输出文件夹路径，保存转换后的 PNG 文件
    """

    subfiles = os.listdir(input_folder_0)
    subfiles.sort()
    count = 0
    for subfile_id, subfile in enumerate(subfiles):
        if 'mask' not in subfile:
            files = os.listdir(os.path.join(input_folder_0, subfile))
            files.sort()

            img_l = []
            for file_id, file in enumerate(files[:5]):

                input_path = os.path.join(input_folder_0, subfile, file)

                img = io.imread(input_path)
                img_array = np.array(img)

                img_l.append(img_array)

            img_array = np.stack(img_l, axis=2)
            img_array = np.max(img_array, axis=2)

            os.makedirs(os.path.join(output_folder), exist_ok=True)

            output_folder_tmp = os.path.join(output_folder, str(count).zfill(4) + '.png')
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
            cv2.imwrite(output_folder_tmp, img_array)

            count += 1

    # subfiles = os.listdir(input_folder_1)
    # subfiles.sort()
    # for subfile_id, subfile in enumerate(subfiles):
    #     if 'mask' not in subfile:
    #         files = os.listdir(os.path.join(input_folder_1, subfile))
    #         files.sort()

    #         img_l = []
    #         for file_id, file in enumerate(files[:5]):

    #             input_path = os.path.join(input_folder_1, subfile, file)

    #             img = io.imread(input_path)
    #             img_array = np.array(img)

    #             img_l.append(img_array)

    #         img_array = np.stack(img_l, axis=2)
    #         img_array = np.max(img_array, axis=2)

    #         os.makedirs(os.path.join(output_folder), exist_ok=True)
    #         output_folder_tmp = os.path.join(output_folder, str(count).zfill(4) + '.png')
    #         img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
    #         cv2.imwrite(output_folder_tmp, img_array)

    #         count += 1

input_folder_0 = '../detection/data/unmixing/datasets-tif/'
input_folder_1 = None
output_folder = '../detection/data/unmixing/AllData/'
output_folder_visualize = '../detection/data/unmixing/AllData_ori/'

ori_augmentation(input_folder_0, input_folder_1, output_folder)
weighted_synthesis(input_folder_0, input_folder_1, output_folder)
visualize(input_folder_0, input_folder_1, output_folder_visualize)






