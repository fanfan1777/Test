
import os
import random
import numpy as np
import torch
from skimage.measure import label, regionprops, find_contours
from sklearn.utils import shuffle
from utils.metrics import precision, recall, F2, dice_score, jac_score  # Fix missing package import when running as module.
from sklearn.metrics import accuracy_score
import cv2
import numpy as np


def resize_keep_aspect_ratio(image, target_size, value=0):
    """
    等比例缩放并填充黑边
    :param image: 输入图像 (H, W) 或 (H, W, C)
    :param target_size: 目标尺寸 tuple (H, W)，例如 (256, 256)
    :param value: 填充颜色，默认黑色 0
    :return: 调整后的图像
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # 计算缩放比例，取最小比例以保证能完全放入
    scale = min(target_w / w, target_h / h)

    # 计算新的宽和高
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 进行等比例缩放
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 创建目标画布
    if len(image.shape) == 3:  # 彩色图/三通道图
        new_image = np.full((target_h, target_w, image.shape[2]), value, dtype=image.dtype)
        # 将缩放后的图贴到画布中心 (或者是左上角，看习惯)
        # 这里演示贴到左上角，计算方便；也可以贴中间
        new_image[:new_h, :new_w, :] = resized
    else:  # 灰度图/Mask
        new_image = np.full((target_h, target_w), value, dtype=image.dtype)
        new_image[:new_h, :new_w] = resized

    return new_image

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Shuffle the dataset. """
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")


""" Convert a mask to border image """
def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

""" Mask to bounding boxes """
def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes

def calculate_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    ## Score
    score_jaccard = jac_score(y_true, y_pred)
    score_f1 = dice_score(y_true, y_pred)
    score_recall = recall(y_true, y_pred)
    score_precision = precision(y_true, y_pred)
    score_fbeta = F2(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_fbeta]



