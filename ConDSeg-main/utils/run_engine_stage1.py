import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
import kornia.augmentation as K
from utils.utils import calculate_metrics,compute_sdf # Fix module import path for stage1 engine.

from utils.metrics import UtralSoundConsistencyLoss,SDFConsistencyLoss
from tqdm import tqdm
import torch.nn.functional as F

#
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn.functional as F


_SUPPORTED_EXTENSIONS = (
    ".npy",
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


def _resize_with_padding(img, target_size, is_mask=False):
    """Resize while keeping aspect ratio by padding the shorter side (letterbox)."""
    target_h, target_w = target_size
    h, w = img.shape[:2]
    if (h, w) == (target_h, target_w):
        return img

    scale = min(target_h / h, target_w / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    border_value = 0 if img.ndim == 2 else (0, 0, 0)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_value)
    return padded


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        # print("n_samples:", self.n_samples)
        # self.convert_edge=convert_edge
        self.size = size

    def __getitem__(self, index):
        # 1. 读取图像 (H, W, 3) - float32 [0, 1]
        # 内容是 [Raw, SASAD, PC]
        image = np.load(self.images_path[index])

        # 2. 读取掩码 (H, W) - uint8 [0, 255]
        mask = np.load(self.masks_path[index])

        # 构造背景通道 (0-255)
        background = 255 - mask

        """ Applying Data Augmentation """
        if self.transform is not None:
            # Albumentations 支持 numpy array 输入
            augmentations = self.transform(image=image, mask=mask, background=background)
            image = augmentations["image"]
            mask = augmentations["mask"]
            background = augmentations["background"]

        """ Resize """
        # 使用之前定义的 _resize_with_padding (需确保该函数能处理 float 输入)
        image = _resize_with_padding(image, self.size)

        # 图像处理：转置 (H, W, 3) -> (3, H, W)
        image = np.transpose(image, (2, 0, 1))
        # 注意：预处理时已经是 [0,1] 的 float32，这里不需要再除以 255.0
        # 如果你使用了其他归一化方式，请在这里检查。
        # image = image / 255.0  <-- 删除这一行，因为 load 进来已经是 0-1 float32
        """ Mask Processing """
        #TODO修改
        mask_binary = (mask > 127).astype(np.float32)
        sdf_gt = compute_sdf(mask_binary)

        mask = _resize_with_padding(mask, self.size, is_mask=True)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0  # Mask 原来是 uint8 0-255，这里需要除以 255

        #TODO修改
        sdf_gt = _resize_with_padding(sdf_gt, self.size, is_mask=False)
        sdf_gt = np.expand_dims(sdf_gt, axis=0)
        """ Background Processing """
        background = _resize_with_padding(background, self.size, is_mask=True)
        background = np.expand_dims(background, axis=0)
        background = background / 255.0

        return image, (mask, background,sdf_gt)

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


# class AddMultiplicativeNoise(nn.Module):
#     def __init__(self, multiplier_range=(0.85, 1.15), elementwise=True, p=0.5):
#         super().__init__()
#         self.min_m, self.max_m = multiplier_range
#         self.elementwise = elementwise
#         self.p = p
#
#     def forward(self, x):
#         # x: (Batch, Channel, Height, Width)
#         if torch.rand(1).item() < self.p:
#             # 生成随机因子
#             if self.elementwise:
#                 # 每个像素通过不同的因子相乘 (模拟斑点噪声 Speckle Noise)
#                 noise = torch.rand_like(x) * (self.max_m - self.min_m) + self.min_m
#             else:
#                 # 整张图乘以同一个因子 (模拟整体增益 Gain 变化)
#                 # 生成 (B, C, 1, 1) 的张量以便广播
#                 B, C, H, W = x.shape
#                 noise = torch.rand(B, C, 1, 1, device=x.device) * (self.max_m - self.min_m) + self.min_m
#
#             return x * noise
#         return x
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}(range={self.min_m}-{self.max_m}, p={self.p})"

# def train(model, loader, optimizer, loss_fn, device, consistency_loss_fn=BinaryConsistencyLoss()):
#     model.train()
#
#     epoch_loss = 0.0
#     epoch_jac = 0.0
#     epoch_f1 = 0.0
#     epoch_recall = 0.0
#     epoch_precision = 0.0
#
#     augmentations = nn.Sequential(
#         # 亮度、对比度调整 (不改变几何位置)
#         K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0, p=0.8),
#         # scale控制遮挡面积比例，ratio控制长宽比
#         K.RandomErasing(scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0.0, p=0.5),
#         # gamma < 1 图像变亮/变白（模拟低动态范围）
#         # gamma > 1 图像变暗/变黑（模拟高对比度）
#         K.RandomGamma(gamma=(0.4, 1.5), gain=(0.8, 1.2), p=0.5),
#         # roughness 控制云雾的粗糙度，shade_intensity 控制遮挡的深度
#         K.RandomPlasmaShadow(roughness=(0.1, 0.5), shade_intensity=(-0.6, 0.0), p=0.4)
#     )
#
#     for i, ((x), (y1, y2)) in enumerate(tqdm(loader, desc="Training", total=len(loader))):
#
#         x = x.to(device, dtype=torch.float32)
#         y1 = y1.to(device, dtype=torch.float32)
#         y2 = y2.to(device, dtype=torch.float32)
#
#         optimizer.zero_grad()
#
#
#         mask_pred = model(x)
#
#         x_aug = augmentations(x)
#         x_aug = x_aug.to(device, dtype=torch.float32)
#         mask_pred_aug = model(x_aug)
#
#         loss_consistency = consistency_loss_fn(mask_pred, mask_pred_aug)
#
#         loss_mask = loss_fn(mask_pred, y1)
#         loss_mask_aug = loss_fn(mask_pred_aug, y1)
#
#         loss = loss_mask + loss_mask_aug + loss_consistency
#
#         loss.backward()
#
#         optimizer.step()
#         epoch_loss += loss.item()
#
#         """ Calculate the metrics """
#         batch_jac = []
#         batch_f1 = []
#         batch_recall = []
#         batch_precision = []
#
#         for yt, yp in zip(y1, mask_pred):
#             score = calculate_metrics(yt, yp)
#             batch_jac.append(score[0])
#             batch_f1.append(score[1])
#             batch_recall.append(score[2])
#             batch_precision.append(score[3])
#
#         epoch_jac += np.mean(batch_jac)
#         epoch_f1 += np.mean(batch_f1)
#         epoch_recall += np.mean(batch_recall)
#         epoch_precision += np.mean(batch_precision)
#
#     epoch_loss = epoch_loss / len(loader)
#     epoch_jac = epoch_jac / len(loader)
#     epoch_f1 = epoch_f1 / len(loader)
#     epoch_recall = epoch_recall / len(loader)
#     epoch_precision = epoch_precision / len(loader)
#
#     return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]
def train(model, loader, optimizer, loss_fn, device, consistency_loss_fn=None, consistency_weight=1.0):
    """
    修改说明:
    1. 默认使用 UtralSoundConsistencyLoss 替代 BinaryConsistencyLoss。
    2. 增加了 consistency_weight 参数用于调节一致性损失的权重。
    3. 显式处理 Logits -> Probabilities 的转换。
    """
    # 如果没有指定一致性损失函数，默认使用模块2的 UtralSoundConsistencyLoss
    if consistency_loss_fn is None:
        consistency_loss_fn = UtralSoundConsistencyLoss().to(device)

    # 新增: SDF 一致性 Loss
    sdf_consistency_loss_fn = SDFConsistencyLoss(weight=1.0).to(device)
    # 新增: SDF 监督 Loss (MSE)
    sdf_supervised_loss_fn = nn.MSELoss().to(device)
    model.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    # 数据增强定义 (保持不变)
    augmentations = nn.Sequential(
        K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0, p=0.8),
        K.RandomErasing(scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0.0, p=0.5),
        K.RandomGamma(gamma=(0.4, 1.5), gain=(0.8, 1.2), p=0.5),
        K.RandomPlasmaShadow(roughness=(0.1, 0.5), shade_intensity=(-0.6, 0.0), p=0.4)
    )

    for i, ((x), (y1, y2,sdf_gt)) in enumerate(tqdm(loader, desc="Training", total=len(loader))):
        x = x.to(device, dtype=torch.float32)
        y1 = y1.to(device, dtype=torch.float32)
        y2 = y2.to(device, dtype=torch.float32)
        #TODO 修改
        sdf_gt = sdf_gt.to(device, dtype=torch.float32)  # 新增: SDF GT

        optimizer.zero_grad()

        # 1. 前向传播 (原始图像)
        # 假设 model 输出的是 Logits (未经过 Sigmoid)
        # mask_pred_logits = model(x)
        #TODO 修改
        prob_pred, sdf_pred = model(x)
        # 2. 前向传播 (增强图像)
        x_aug = augmentations(x)
        x_aug = x_aug.to(device, dtype=torch.float32)
        # 假设 model 输出的是 Logits
        # mask_pred_aug_logits = model(x_aug)
        prob_pred_aug, sdf_pred_aug = model(x_aug)
        # 3. 概率转换 (Logits -> Probabilities)
        # 关键: KL散度、FFT一致性和DiceLoss都需要在 [0,1] 概率空间计算
        # prob_pred = torch.sigmoid(mask_pred_logits)
        # prob_pred_aug = torch.sigmoid(mask_pred_aug_logits)
        # prob_pred = mask_pred_logits
        # prob_pred_aug = mask_pred_aug_logits
        # 4. 计算一致性损失 (Module 2: Soft Structure Consistency)
        # 使用概率图作为输入
        loss_consistency = consistency_loss_fn(prob_pred, prob_pred_aug)

        # 5. 计算监督损失 (DiceBCELoss)
        # 注意: 如果 mask_pred 是 Logits，这里必须传入处理后的 prob_pred，
        # 因为 utils/metrics.py 中的 DiceBCELoss 期望输入是概率。
        loss_mask = loss_fn(prob_pred, y1)

        # 也可以计算增强后的监督损失 (可选，保持原逻辑)
        loss_mask_aug = loss_fn(prob_pred_aug, y1)

        # B. 概率图一致性 Loss (Module 2)
        loss_prob_cons = consistency_loss_fn(prob_pred, prob_pred_aug)

        # C. SDF 监督 Loss (关键! 必须有这个网络才能学会输出 SDF)
        # 只需要对原始图像计算监督，增强图像可选
        loss_sdf_sup = sdf_supervised_loss_fn(sdf_pred, sdf_gt)

        #TODO修改 SDF 一致性 Loss (方案A核心)
        loss_sdf_cons = sdf_consistency_loss_fn(sdf_pred, sdf_pred_aug)
        # 6. 总损失
        # loss = L_supervised + lambda * L_consistency
        # loss = loss_mask + loss_mask_aug + (consistency_weight * loss_consistency)
        lambda_sdf = 0.5
        loss = loss_mask + loss_mask_aug + (consistency_weight * loss_consistency) +lambda_sdf * (loss_sdf_sup + loss_sdf_cons)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        """ Calculate the metrics """
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        # 计算指标时也使用概率图 prob_pred
        for yt, yp in zip(y1, prob_pred):
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
        for i, ((x), (y1, y2,sdf_gt)) in enumerate(tqdm(loader, desc="Evaluation", total=len(loader))):
            x = x.to(device, dtype=torch.float32)
            y1 = y1.to(device, dtype=torch.float32)
            y2 = y2.to(device, dtype=torch.float32)
            
            mask_pred,sdf_pred = model(x)

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
