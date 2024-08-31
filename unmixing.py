import os
import cv2
import numpy as np
from PIL import Image
import time
from skimage import io
import utility


sample_id_l = [5, 15, 18, 19, 25, 26, 20, 27, 24]

for sample_id in sample_id_l:
    sample_id = sample_id + 1

    tif_path = 'detection/data/unmixing/fluorescence_time_data_tif/' + str(sample_id).zfill(4) + '/'
    tif_files_all = os.listdir(tif_path)
    tif_files_all.sort()
    save_path = 'unmix/results_' + str(sample_id).zfill(2) + '/'
    os.makedirs(save_path, exist_ok=True)

    # path = '/opt/cr/unmixing/yolov5-master/results_unmixing10_2_auto_time04_crop3'
    path = 'metric/results_' +str(sample_id).zfill(2)
    files_all = os.listdir(path)
    files_all.sort()


    n = files_all[-1]
    n = int(n.split('_')[0])

    name_data_l = []
    for i in range(1, n):
        name_data_l.append(str(i).zfill(4))
    # name_data_l = ['0006', '0016', '0017', '0018']
    name_class_l = ['tumor', 'intestine', 'colon', 'lymph', 'vessel']

    A_l_past = []
    for idx, name_data in enumerate(name_data_l):
        A_l = []
        for name_class in name_class_l:
            if os.path.exists(os.path.join(path, name_data + '_' + name_class + '_mask.png')) == False:
                continue

            mask = cv2.imread(os.path.join(path, name_data + '_' + name_class + '_mask.png'), cv2.IMREAD_GRAYSCALE)

            mask = mask/255.
            A_l_class = []
            for i in range(5):
                tif = io.imread(os.path.join(path, name_data + '_' + name_class + '_' + str(i).zfill(2) + '.tif'))
                tif = np.array(tif)
                tif = tif * mask

                height, width = tif.shape
                border_width = int(0.2 * width)
                border_height = int(0.2 * height)

                tif = tif[border_height:-border_height, border_width:-border_width]
                mask_copy = mask[border_height:-border_height, border_width:-border_width].copy()
                A_l_class.append(np.sum(tif) / (np.sum(mask_copy) + 1e-4) )

            A_l.append(A_l_class)
        if len(A_l) != 5:
            continue
        # for A in A_l:
        #     print(A)
        A_l = np.array(A_l)
        A_l = A_l.T
        # print(A_l)
        # print('A_l w/o average:', A_l)

        A_l_past.append(A_l)

        if len(A_l_past) > 32:
            del (A_l_past[0])


        if len(A_l_past) >= 3:
            data_all = np.array(A_l_past)
            data = np.array(A_l_past)

            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            # 计算 Z-score
            z_scores = np.abs((data - mean) / (std + 1e-6))

            z_scores = np.mean(z_scores.reshape(-1, 25), axis=-1)

            threshold = np.sort(z_scores.ravel())[len(z_scores.ravel()) // 2]

            filtered_data = data_all[z_scores - threshold <= 1e-4]
            filtered_data = np.mean(filtered_data, axis=0)

            A_l = filtered_data
        # print('idx', idx)
        # print('A_l average:', A_l)

        tif_files = tif_files_all[idx * 5:(idx + 1) * 5]

        tif_l = []
        for tif_file in tif_files:
            input_path = os.path.join(tif_path, tif_file)
            tif = io.imread(input_path)
            tif_array = np.array(tif)
            tif_l.append(tif_array)
        tif_l = np.stack(tif_l, axis=2)
        colored_img = np.max(tif_l, axis = -1)
        colored_img = np.stack([colored_img]*3, axis=-1)

        unmixing_part = ['tumor', 'intestine', 'colon', 'lymph', 'vessel']
        h, w, c = tif_l.shape
        _, n = A_l.shape
        tif_l = tif_l.reshape(h * w, c)
        unminxing_results = (np.linalg.pinv(A_l) @ tif_l.T).T
        unminxing_results = unminxing_results.reshape(h, w, n)
        unminxing_results[unminxing_results < 0] = 0

        color_l = [[0, 0, 255],
                   [255, 0, 0],
                   [255, 0, 255],
                   [0, 255, 0],
                   [255, 255, 0],
                   ]

        visualization = np.zeros((unminxing_results.shape[0], unminxing_results.shape[1], 3), dtype=np.uint8)
        for i in range(unminxing_results.shape[2]):

            tif = unminxing_results[:, :, i]

            unmixing_save_path = os.path.join(save_path, str(idx).zfill(4) + '_unmixing_' + unmixing_part[i] + '.tif')
            utility.save_tiff_imagej_compatible(unmixing_save_path, tif.astype(np.float32), 'YX')

            tif = (tif - tif.min()) / (tif.max() - tif.min()) * 255

            unmixing_save_path = os.path.join(save_path, str(idx).zfill(4) + '_unmixing_' + unmixing_part[i] + '.png')
            cv2.imwrite(unmixing_save_path, tif)

            tif_normalized = tif.astype(np.uint8)

            # Create a color mask
            color_mask = np.zeros_like(visualization)
            for j in range(3):
                color_mask[:, :, j] = tif_normalized * (color_l[i][j] / 255)

            # Add the color mask to the visualization image
            visualization = cv2.add(visualization, color_mask)

        unmixing_save_path = os.path.join(save_path, str(idx).zfill(4) + '_unmixing_all' + '.png')
        cv2.imwrite(unmixing_save_path, visualization)

        exit(0)






