import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.util import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics, mask_to_bbox
from tqdm import tqdm


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn.functional as F


_SUPPORTED_EXTENSIONS = (
    ".jpg", ".JPG", ".jpeg", ".JPEG",
    ".png", ".PNG", ".bmp", ".BMP",
    ".tif", ".tiff", ".TIF", ".TIFF",
)


def _resolve_file(path, folder, name):
    """Resolve file names saved without extensions inside txt protocol splits."""
    base = os.path.join(path, folder, name)
    for ext in _SUPPORTED_EXTENSIONS:
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find {name} with supported extensions under {folder}")


def _gather_files(folder):
    files = []
    for ext in _SUPPORTED_EXTENSIONS:
        files.extend(sorted(glob(os.path.join(folder, f"*{ext}"))))
    return files


def _collect_pairs_from_dirs(image_dir, mask_dir):
    if not (os.path.isdir(image_dir) and os.path.isdir(mask_dir)):
        return None

    image_files = _gather_files(image_dir)
    mask_files = _gather_files(mask_dir)
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found inside {image_dir}")

    mask_lookup = {os.path.splitext(os.path.basename(m))[0]: m for m in mask_files}
    images, masks = [], []
    missing = []
    for img in image_files:
        stem = os.path.splitext(os.path.basename(img))[0]
        mask_path = mask_lookup.get(stem)
        if mask_path is None:
            missing.append(stem)
            continue
        images.append(img)
        masks.append(mask_path)

    if missing:
        raise FileNotFoundError(
            f"Missing masks for {len(missing)} samples under {mask_dir}. Examples: {missing[:5]}"
        )

    return images, masks


def _split_pairs(images, masks, val_split, seed):
    if len(images) != len(masks):
        raise ValueError("Image/mask counts do not match")
    if len(images) < 2:
        raise ValueError("Need at least two samples to create a validation split")
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(images))
    rng.shuffle(indices)

    val_count = max(1, int(len(indices) * val_split))
    if val_count >= len(indices):
        val_count = len(indices) - 1

    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    def _take(idxs):
        return [images[i] for i in idxs], [masks[i] for i in idxs]

    return _take(train_idx), _take(val_idx)


def _load_from_directory_structure(path, val_split, seed, use_test_split):
    train_image_dir = os.path.join(path, "trainval-image")
    train_mask_dir = os.path.join(path, "trainval-mask")

    if not (os.path.isdir(train_image_dir) and os.path.isdir(train_mask_dir)):
        return None

    train_images, train_masks = _collect_pairs_from_dirs(train_image_dir, train_mask_dir)

    if use_test_split:
        test_image_dir = os.path.join(path, "test-image")
        test_mask_dir = os.path.join(path, "test-mask")
        if os.path.isdir(test_image_dir) and os.path.isdir(test_mask_dir):
            test_pairs = _collect_pairs_from_dirs(test_image_dir, test_mask_dir)
            return (train_images, train_masks), test_pairs

    (train_split, val_split_pairs) = _split_pairs(train_images, train_masks, val_split, seed)
    return train_split, val_split_pairs


def load_names(path, file_path):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [_resolve_file(path, "images", name) for name in data]
    masks = [_resolve_file(path, "masks", name) for name in data]
    return images, masks


def load_data(path, val_name=None, val_split=0.1, seed=42, use_test_split=False):
    train_names_path = f"{path}/train.txt"
    if os.path.exists(train_names_path):
        if use_test_split and os.path.exists(f"{path}/test.txt"):
            test_names_path = f"{path}/test.txt"
            train_x, train_y = load_names(path, train_names_path)
            test_x, test_y = load_names(path, test_names_path)
            return (train_x, train_y), (test_x, test_y)

        if val_name is None:
            valid_names_path = f"{path}/val.txt"
        else:
            valid_names_path = f"{path}/val_{val_name}.txt"

        train_x, train_y = load_names(path, train_names_path)
        valid_x, valid_y = load_names(path, valid_names_path)
        return (train_x, train_y), (valid_x, valid_y)

    folder_split = _load_from_directory_structure(path, val_split, seed, use_test_split)
    if folder_split is not None:
        return folder_split

    raise FileNotFoundError(
        "Unable to locate supported split files or folders. Expected train.txt/val.txt or trainval-image/trainval-mask directories."
    )

class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        self.size = size

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

        """ Image """
        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0

        """ Mask """
        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0

        """ Background """
        background = cv2.resize(background, self.size)
        background = np.expand_dims(background, axis=0)
        background = background / 255.0

        return image, (mask, background)

    def __len__(self):
        return self.n_samples


def complementary_loss(prob_fg, prob_bg, prob_uc):
    loss = (prob_fg * prob_bg).sum() + (prob_fg * prob_uc).sum() + (prob_bg * prob_uc).sum()
    num_pixels = prob_fg.size(0) * prob_fg.size(2) * prob_fg.size(3)  # B * H * W
    normalized_loss = loss / num_pixels
    return normalized_loss

def train(model, loader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    for i, ((x), (y1, y2)) in enumerate(tqdm(loader, desc="Training", total=len(loader))):
        x = x.to(device, dtype=torch.float32)
        y1 = y1.to(device, dtype=torch.float32)
        y2 = y2.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        mask_pred, fg_pred, bg_pred, uc_pred = model(x)

        loss_mask = loss_fn(mask_pred, y1)
        loss_fg = loss_fn(fg_pred, y1)
        loss_bg = loss_fn(bg_pred, y2)

        beta1 = 1 / (torch.tanh(fg_pred.sum() / (fg_pred.shape[2] * fg_pred.shape[3])) + 1e-15)
        beta2 = 1 / (torch.tanh(bg_pred.sum() / (bg_pred.shape[2] * bg_pred.shape[3])) + 1e-15)
        beta1 = beta1.to(device)
        beta2 = beta2.to(device)
        preds = torch.stack([fg_pred, bg_pred, uc_pred], dim=1)
        probs = F.softmax(preds, dim=1)
        prob_fg, prob_bg, prob_uc = probs[:, 0], probs[:, 1], probs[:, 2]

        loss_comp = complementary_loss(prob_fg, prob_bg, prob_uc)
        loss_comp = loss_comp.to(device)
        loss = loss_mask + beta1 * loss_fg + beta2 * loss_bg + loss_comp
        loss.backward()

        optimizer.step()
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


def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for i, ((x), (y1, y2)) in enumerate(tqdm(loader, desc="Evaluation", total=len(loader))):
            x = x.to(device, dtype=torch.float32)
            y1 = y1.to(device, dtype=torch.float32)
            y2 = y2.to(device, dtype=torch.float32)

            mask_pred, fg_pred, bg_pred, uc_pred = model(x)

            loss_mask = loss_fn(mask_pred, y1)
            loss_fg = loss_fn(fg_pred, y1)
            loss_bg = loss_fn(bg_pred, y2)

            beta1 = 1 / (torch.tanh(fg_pred.sum() / (fg_pred.shape[2] * fg_pred.shape[3])) + 1e-15)
            beta2 = 1 / (torch.tanh(bg_pred.sum() / (bg_pred.shape[2] * bg_pred.shape[3])) + 1e-15)
            beta1 = beta1.to(device)
            beta2 = beta2.to(device)

            preds = torch.stack([fg_pred, bg_pred, uc_pred], dim=1)
            probs = F.softmax(preds, dim=1)
            prob_fg, prob_bg, prob_uc = probs[:, 0], probs[:, 1], probs[:, 2]

            loss_comp = complementary_loss(prob_fg, prob_bg, prob_uc)
            loss_comp = loss_comp.to(device)

            loss = loss_mask + beta1 * loss_fg + beta2 * loss_bg + loss_comp

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
