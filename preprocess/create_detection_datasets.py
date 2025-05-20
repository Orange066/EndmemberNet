import os
from PIL import Image
import cv2 as cv2
import numpy as np
import shutil

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
box_visualization_path = '../detection/data/unmixing/box_visualization/'
os.makedirs(box_visualization_path, exist_ok=True)
save_label_path = '../detection/data/unmixing/labels/'
os.makedirs(save_label_path, exist_ok=True)
yaml_file = '../detection/data/unmixing/unmixing_supplementary.yaml'

augment_img_path = '../detection/data/unmixing/AllData/'
img_save_path = '../detection/data/unmixing/images/'
os.makedirs(img_save_path, exist_ok=True)

# create txt, yaml
for i in range(len(files)):

    # create augment img
    shutil.copyfile(os.path.join(path, files[i]), os.path.join(img_save_path, files[i]))
    shutil.copyfile(os.path.join(path.replace('/_rgb/', '/_rgb_eqh/'), files[i]), os.path.join(img_save_path, files[i].replace('.png', '_eqh.png')))
    augment_imgs = os.listdir(os.path.join(augment_img_path, str(i).zfill(4)))
    augment_imgs.sort()
    for j, augment_img in enumerate(augment_imgs):
        shutil.copyfile(os.path.join(augment_img_path, str(i).zfill(4), augment_img), os.path.join(img_save_path, files[i].replace('.png', '_'+str(j).zfill(4)+'.png')))


train_file = open(os.path.join(save_txt_path, 'train.txt'), 'w')
test_file = open(os.path.join(save_txt_path, 'test.txt'), 'w')
val_file = open(os.path.join(save_txt_path, 'val.txt'), 'w')

select_test = [5, 15, 18, 19, 25, 26, 29, 30, 31, 32]
select_valid = [20, 27, 24]
for j in range(len(files)):
    img_path = os.path.join(img_save_path, files[j])
    img_eqh_path = os.path.join(img_save_path, files[j].replace('.png', '_eqh.png'))
    if j in select_test:
        test_file.write(img_path + '\n')
        test_file.write(img_eqh_path + '\n')

        augment_imgs = os.listdir(os.path.join(augment_img_path, str(i).zfill(4)))
        augment_imgs.sort()
        for k, augment_img in enumerate(augment_imgs):
            test_file.write(
                os.path.join(img_save_path, files[j].replace('.png', '_' + str(k).zfill(4) + '.png')) + '\n')
    elif j in select_valid:
        val_file.write(img_path + '\n')
        val_file.write(img_eqh_path + '\n')

        augment_imgs = os.listdir(os.path.join(augment_img_path, str(i).zfill(4)))
        augment_imgs.sort()
        for k, augment_img in enumerate(augment_imgs):
            val_file.write(
                os.path.join(img_save_path, files[j].replace('.png', '_' + str(k).zfill(4) + '.png')) + '\n')
    else:
        # test_file.write(img_path + '\n')
        # val_file.write(img_path + '\n')
        train_file.write(img_path + '\n')
        train_file.write(img_eqh_path + '\n')

        augment_imgs = os.listdir(os.path.join(augment_img_path, str(i).zfill(4)))
        augment_imgs.sort()
        for k, augment_img in enumerate(augment_imgs):
            train_file.write(os.path.join(img_save_path, files[j].replace('.png', '_' + str(k).zfill(4) + '.png')) + '\n')


train_file.close()
test_file.close()
val_file.close()

# create label

mask_class = os.listdir(mask_path)
mask_class.sort()
# mask_class = mask_class[1:]
mask_files = os.listdir(os.path.join(mask_path, mask_class[0]))
mask_files.sort()

for i, file in enumerate(mask_files):
    ori_image = cv2.imread(os.path.join(path, files[i]), cv2.IMREAD_GRAYSCALE)
    label_file = open(os.path.join(save_label_path, files[i].replace('.png', '.txt')), 'w')
    label_eqh_file = open(os.path.join(save_label_path, files[i].replace('.png', '_eqh.txt')), 'w')

    augment_imgs = os.listdir(os.path.join(augment_img_path, str(i).zfill(4)))
    augment_imgs.sort()
    augment_label_files = []
    for k, augment_img in enumerate(augment_imgs):
        augment_label_file = open(os.path.join(save_label_path, files[i].replace('.png', '_'+str(k).zfill(4)+'.txt')), 'w')
        augment_label_files.append(augment_label_file)

    for i_c, mask_c in enumerate(mask_class):
        mask = cv2.imread(os.path.join(mask_path, mask_c, file), cv2.IMREAD_GRAYSCALE)

        if np.sum(mask) < 1:
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1:
            print('Waring! len(contours) != 1')
            exit(0)

        # 计算每个轮廓的边界框
        x, y, w, h = cv2.boundingRect(contours[0])

        # 可选：在一个空白图像上绘制边界框
        cv2.rectangle(ori_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        size = [ori_image.shape[1], ori_image.shape[0]]
        box = [x, y, x + w, y + h]
        x, y, w, h = convert(size, box)
        label_file.write(str(i_c) + " " + " ".join([str(a) for a in [x, y, w, h]]) + '\n')
        label_eqh_file.write(str(i_c) + " " + " ".join([str(a) for a in [x, y, w, h]]) + '\n')

        for augment_label in augment_label_files:
            augment_label.write(str(i_c) + " " + " ".join([str(a) for a in [x, y, w, h]]) + '\n')

    label_file.close()
    label_eqh_file.close()
    for augment_label in augment_label_files:
        augment_label.close()
    cv2.imwrite(os.path.join(box_visualization_path, file), ori_image)


