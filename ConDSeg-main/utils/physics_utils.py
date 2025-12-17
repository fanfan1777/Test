import numpy as np
import cv2

# 务必确保安装了 phasepack: pip install phasepack
try:
    from phasepack import phasecong
except ImportError:
    raise ImportError("请先安装 phasepack 库: pip install phasepack")


def _normalize(channel: np.ndarray) -> np.ndarray:
    """Normalize a 2D array to [0, 1] with numerical stability."""
    channel = channel.astype(np.float32)
    min_val = float(channel.min())
    max_val = float(channel.max())
    denom = max(max_val - min_val, 1e-8)
    channel = (channel - min_val) / denom
    return np.clip(channel, 0.0, 1.0)


def compute_sasad(gray: np.ndarray, num_iter: int = 15, delta_t: float = 0.14, q0_factor: float = 1.0) -> np.ndarray:
    """
    SASAD implementation.
    Args:
        gray: HxW numpy array (0-255 or 0-1)
    """
    # 1. 初始类型转换
    img = gray.astype(np.float32) / 255.0 if gray.dtype == np.uint8 else gray.astype(np.float32)
    eps = 1e-10

    for t in range(num_iter):
        # 使用 symmetric 填充边界
        img_pad = np.pad(img, ((1, 1), (1, 1)), mode='symmetric')
        center = img_pad[1:-1, 1:-1]

        # 2. 计算梯度
        dN = img_pad[0:-2, 1:-1] - center
        dS = img_pad[2:, 1:-1] - center
        dE = img_pad[1:-1, 2:] - center
        dW = img_pad[1:-1, 0:-2] - center

        # 3. 计算瞬时变异系数 q
        grad_mag = np.sqrt(dN ** 2 + dS ** 2 + dE ** 2 + dW ** 2 + eps)
        local_mean = cv2.GaussianBlur(img, (5, 5), 1.0) + eps
        q = grad_mag / local_mean

        # 4. 估计散斑基准值 q0
        q0 = np.median(q) * q0_factor
        if q0 == 0: q0 = eps

        # 5. 计算扩散系数 c(q)
        denom = (q ** 2 - q0 ** 2) / (q0 ** 2 * (1 + q0 ** 2) + eps)
        c = 1.0 / (1.0 + np.maximum(denom, 0))

        # 6. 更新图像
        img += delta_t * (c * dN + c * dS + c * dE + c * dW)

    return _normalize(img)


def compute_phase_congruency(gray: np.ndarray) -> np.ndarray:
    """Generate a phase congruency response map."""
    gray_norm = gray.astype(np.float32) / 255.0
    # nscale=4, norient=6 是常用参数，可根据需求调整
    pc, *_ = phasecong(gray_norm, nscale=4, norient=6)
    pc = np.nan_to_num(pc, nan=0.0, posinf=0.0, neginf=0.0)
    return _normalize(pc)


def build_triplet_tensor(image_bgr: np.ndarray) -> np.ndarray:
    """
    Build the advanced three-channel tensor (Original + SASAD + PC).
    Args:
        image_bgr: BGR image (H, W, 3) read by cv2.imread
    Returns:
        np.ndarray shaped (H, W, 3) (Note: Channels last for easier augmentation later)
    """
    # 转为灰度图用于计算
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 1. 原始图 (归一化)
    original = gray.astype(np.float32) / 255.0

    # 2. SASAD
    sasad = compute_sasad(gray)

    # 3. Phase Congruency
    pc = compute_phase_congruency(gray)

    # 堆叠: 保持 (H, W, 3) 格式，方便后续 Albumentations 处理
    # 注意：你的原代码是 (3, H, W)，但通常预处理存文件建议存 (H, W, 3)，
    # 因为 Albumentations 和 matplotlib 默认处理 HWC。
    # 之后在 Dataset 的 __getitem__ 里再转回 CHW。
    stacked = np.stack([original, sasad, pc], axis=-1).astype(np.float32)

    return stacked