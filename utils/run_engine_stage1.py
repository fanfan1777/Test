import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utils.utils import calculate_metrics  # Fix module import path for stage1 engine.
from utils.preprocessing import build_triplet_tensor, degrade_triplet
from utils.data_io import load_data
import utils
from tqdm import tqdm

#
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn.functional as F
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

class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None, dual_input=True):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        # print("n_samples:", self.n_samples)
        # self.convert_edge=convert_edge
        self.size = size
        self.dual_input = dual_input

    def __getitem__(self, index):
        """ Reading Image & Mask """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        background = mask.copy()
        background = 255 - background
        # Fix BGR/RGB mismatch: align OpenCV's BGR output with RGB expectation before normalization.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        """ Applying Data Augmentation """
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask, background=background)
            image = augmentations["image"]
            mask = augmentations["mask"]
            background = augmentations["background"]

        """ Image -> Advanced triplet tensor """
        image = resize_keep_aspect_ratio(image, self.size)
        primary = build_triplet_tensor(image)
        degraded = degrade_triplet(primary) if self.dual_input else primary.copy()

        """ Mask """
        mask = resize_keep_aspect_ratio(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0

        """ Background """
        background = resize_keep_aspect_ratio(background, self.size)
        background = np.expand_dims(background, axis=0)
        background = background / 255.0

        return (primary, degraded), (mask, background)

    def __len__(self):
        return self.n_samples


class BinaryConsistencyLoss(nn.Module):
    def __init__(self):
        super(BinaryConsistencyLoss, self).__init__()

    def forward(self, mask1, mask2):

        mask1_binary = (mask1 > 0.5).float()
        mask2_binary = (mask2 > 0.5).float()

        loss1 = F.binary_cross_entropy(mask1, mask2_binary, reduction='mean')
        loss2 = F.binary_cross_entropy(mask2, mask1_binary, reduction='mean')

        loss = loss1 + loss2

        return loss


def train(model, loader, optimizer, loss_fn, device, consistency_loss_fn=BinaryConsistencyLoss()):
    model.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    for i, ((x_clean, x_degraded), (y1, y2)) in enumerate(tqdm(loader, desc="Training", total=len(loader))):

        x_clean = x_clean.to(device, dtype=torch.float32)
        x_degraded = x_degraded.to(device, dtype=torch.float32)
        y1 = y1.to(device, dtype=torch.float32)
        y2 = y2.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        mask_pred_clean = model(x_clean)
        mask_pred_degraded = model(x_degraded)

        loss_consistency = consistency_loss_fn(mask_pred_clean, mask_pred_degraded)

        loss_mask_clean = loss_fn(mask_pred_clean, y1)
        loss_mask_degraded = loss_fn(mask_pred_degraded, y1)

        loss = loss_mask_clean + loss_mask_degraded + 0.5 * loss_consistency

        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()

        """ Calculate the metrics """
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        for yt, yp in zip(y1, mask_pred_clean):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss = epoch_loss / len(loader)
    epoch_jac = epoch_jac / len(loader)
    epoch_f1 = epoch_f1 / len(loader)
    epoch_recall = epoch_recall / len(loader)
    epoch_precision = epoch_precision / len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for i, ((x_clean, _), (y1, y2)) in enumerate(tqdm(loader, desc="Evaluation", total=len(loader))):
            x_clean = x_clean.to(device, dtype=torch.float32)
            y1 = y1.to(device, dtype=torch.float32)
            y2 = y2.to(device, dtype=torch.float32)

            mask_pred = model(x_clean)

            loss_mask = loss_fn(mask_pred, y1)

            loss = loss_mask

            epoch_loss += loss.item()

            """ Calculate the metrics """
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for yt, yp in zip(y1, mask_pred):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        epoch_loss = epoch_loss / len(loader)
        epoch_jac = epoch_jac / len(loader)
        epoch_f1 = epoch_f1 / len(loader)
        epoch_recall = epoch_recall / len(loader)
        epoch_precision = epoch_precision / len(loader)

        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]
