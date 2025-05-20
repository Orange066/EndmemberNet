import os
from PIL import Image
import cv2 as cv2
import numpy as np
import shutil
import random

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

path = '../detection/data/unmixing/AllData_ori/'
files = os.listdir(path)
files.sort()

save_txt_path = '../detection/data/unmixing/txt/'
os.makedirs(save_txt_path, exist_ok=True)
mask_path = '../detection/data/unmixing/AllMask/'

save_img_path = '../detection/data/unmixing/AllData_ori_patch/'
save_mask_path = '../detection/data/unmixing/AllMask_patch/'

os.makedirs(save_img_path, exist_ok=True)
os.makedirs(save_mask_path, exist_ok=True)

select_test = [5, 15, 18, 19, 25, 26, 29, 30, 31, 32]
select_valid = [20, 27, 24]

# create label

mask_class = os.listdir(mask_path)
mask_class.sort()
# mask_class = mask_class[1:]
mask_files = os.listdir(os.path.join(mask_path, mask_class[0]))
mask_files.sort()

for i, file in enumerate(mask_files):
    ori_image = cv2.imread(os.path.join(path, files[i]), cv2.IMREAD_GRAYSCALE)

    for i_c, mask_c in enumerate(mask_class):
        mask = cv2.imread(os.path.join(mask_path, mask_c, file), cv2.IMREAD_GRAYSCALE)

        if np.sum(mask) < 1:
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1:
            print('Waring! len(contours) != 1')
            exit(0)

        save_subpath = 'train'
        base_num = int(file[:-4])
        if base_num in select_test:
            save_subpath = 'test'
        if base_num in select_valid:
            save_subpath = 'valid'

        x, y, w, h = cv2.boundingRect(contours[0])

        crop_img = ori_image[y:y + h, x:x + w]
        crop_mask = mask[y:y + h, x:x + w]
        os.makedirs(os.path.join(save_img_path, save_subpath), exist_ok=True)
        os.makedirs(os.path.join(save_mask_path, save_subpath), exist_ok=True)
        cv2.imwrite(os.path.join(save_img_path, save_subpath, file[:-4]+ '_' +mask_c + '.png'), crop_img)
        cv2.imwrite(os.path.join(save_mask_path, save_subpath, file[:-4]+ '_' +mask_c + '.png'), crop_mask)

        for j in range(5):
            w_tmp = w - random.randint(0, int(w * 0.4))
            h_tmp = h -  random.randint(0, int(h * 0.4))
            crop_img = ori_image[y:y+h_tmp, x:x+w_tmp]
            crop_mask = mask[y:y+h_tmp, x:x+w_tmp]

            cv2.imwrite(os.path.join(save_img_path, save_subpath, file[:-4]+ '_' +mask_c + '_00_' + str(j) + '.png'), crop_img)
            cv2.imwrite(os.path.join(save_mask_path, save_subpath, file[:-4]+ '_' +mask_c+ '_00_' + str(j) + '.png'), crop_mask)

        for j in range(5):
            w_tmp = w - random.randint(0, int(w * 0.4))
            h_tmp = h -  random.randint(0, int(h * 0.4))
            crop_img = ori_image[y+h-h_tmp :y+h, x:x+w_tmp]
            crop_mask = mask[y+h-h_tmp :y+h, x:x+w_tmp]

            cv2.imwrite(os.path.join(save_img_path, save_subpath, file[:-4] + '_'+mask_c+ '_01_' + str(j) + '.png'), crop_img)
            cv2.imwrite(os.path.join(save_mask_path, save_subpath, file[:-4] + '_' +mask_c + '_01_' + str(j) + '.png'), crop_mask)

        for j in range(5):
            w_tmp = w - random.randint(0, int(w * 0.4))
            h_tmp = h -  random.randint(0, int(h * 0.4))
            crop_img = ori_image[y:y+h_tmp, x+w-w_tmp:x+w]
            crop_mask = mask[y:y+h_tmp, x+w-w_tmp:x+w]

            cv2.imwrite(os.path.join(save_img_path, save_subpath, file[:-4] + '_' +mask_c+ '_02_' + str(j) + '.png'), crop_img)
            cv2.imwrite(os.path.join(save_mask_path, save_subpath, file[:-4] + '_' +mask_c+ '_02_' + str(j) + '.png'), crop_mask)

        for j in range(5):
            w_tmp = w - random.randint(0, int(w * 0.4))
            h_tmp = h -  random.randint(0, int(h * 0.4))
            crop_img = ori_image[y+h-h_tmp :y+h, x+w-w_tmp:x+w]
            crop_mask = mask[y+h-h_tmp :y+h, x+w-w_tmp:x+w]

            cv2.imwrite(os.path.join(save_img_path, save_subpath, file[:-4] + '_' +mask_c+ '_03_' + str(j) + '.png'), crop_img)
            cv2.imwrite(os.path.join(save_mask_path, save_subpath, file[:-4] + '_' +mask_c+ '_03_' + str(j) + '.png'), crop_mask)


def resize_images_cv2(input_dir, output_dir, size=(128, 128), extensions={'.jpg', '.jpeg', '.png', '.bmp'}):
    for root, _, files in os.walk(input_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in extensions:
                input_path = os.path.join(root, file)

                # 构造输出路径，保留子目录结构
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                try:
                    img = cv2.imread(input_path)
                    if img is None:
                        print(f"Skipped invalid image: {input_path}")
                        continue
                    resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(output_path, resized_img)
                    print(f"Saved: {output_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

# 设置路径
input_folder = '../detection/data/unmixing/AllData_ori_patch/'
output_folder = '../detection/data/unmixing/AllData_ori_patch_resize/'
resize_images_cv2(input_folder, output_folder)

input_folder = '../detection/data/unmixing/AllMask_patch/'
output_folder = '../detection/data/unmixing/AllMask_patch_resize/'
resize_images_cv2(input_folder, output_folder)




















