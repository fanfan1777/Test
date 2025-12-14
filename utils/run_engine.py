import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics, mask_to_bbox
from utils.preprocessing import build_triplet_tensor, degrade_triplet
from utils.data_io import load_data
from tqdm import tqdm


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn.functional as F


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None, dual_input=True):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        self.size = size
        self.dual_input = dual_input

    def __getitem__(self, index):
        """ Reading Image & Mask """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        background = mask.copy()
        background = 255 - background
        # Fix BGR/RGB mismatch: convert OpenCV-loaded BGR image to RGB before feeding the model.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        """ Applying Data Augmentation """
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask, background=background)
            image = augmentations["image"]
            mask = augmentations["mask"]
            background = augmentations["background"]

        """ Image -> Advanced triplet tensor """
        image = cv2.resize(image, self.size)
        primary = build_triplet_tensor(image)
        degraded = degrade_triplet(primary) if self.dual_input else primary.copy()

        """ Mask """
        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0

        """ Background """
        background = cv2.resize(background, self.size)
        background = np.expand_dims(background, axis=0)
        background = background / 255.0

        return (primary, degraded), (mask, background)

    def __len__(self):
        return self.n_samples


def complementary_loss(prob_fg, prob_bg, prob_uc):
    loss = (prob_fg * prob_bg).sum() + (prob_fg * prob_uc).sum() + (prob_bg * prob_uc).sum()
    num_pixels = prob_fg.size(0) * prob_fg.size(2) * prob_fg.size(3)  # B * H * W
    normalized_loss = loss / num_pixels
    return normalized_loss


def _forward_branch(model, x, y1, y2, loss_fn):
    mask_pred, fg_pred, bg_pred, uc_pred = model(x)

    loss_mask = loss_fn(mask_pred, y1)
    loss_fg = loss_fn(fg_pred, y1)
    loss_bg = loss_fn(bg_pred, y2)

    spatial_size = fg_pred.shape[2] * fg_pred.shape[3]
    beta1 = 1 / (torch.tanh(fg_pred.sum() / spatial_size) + 1e-15)
    beta2 = 1 / (torch.tanh(bg_pred.sum() / spatial_size) + 1e-15)

    preds = torch.stack([fg_pred, bg_pred, uc_pred], dim=1)
    probs = F.softmax(preds, dim=1)
    prob_fg, prob_bg, prob_uc = probs[:, 0], probs[:, 1], probs[:, 2]
    loss_comp = complementary_loss(prob_fg, prob_bg, prob_uc)

    total_loss = loss_mask + beta1 * loss_fg + beta2 * loss_bg + loss_comp
    return total_loss, mask_pred

def train(model, loader, optimizer, loss_fn, device, secondary_weight: float = 0.5, consistency_weight: float = 0.2):
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

        clean_loss, mask_pred_clean = _forward_branch(model, x_clean, y1, y2, loss_fn)
        degraded_loss, mask_pred_degraded = _forward_branch(model, x_degraded, y1, y2, loss_fn)
        consistency_loss = F.mse_loss(mask_pred_clean, mask_pred_degraded)
        loss = clean_loss + secondary_weight * degraded_loss + consistency_weight * consistency_loss
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

            loss, mask_pred = _forward_branch(model, x_clean, y1, y2, loss_fn)

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
