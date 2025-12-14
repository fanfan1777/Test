import random

import cv2
import numpy as np

try:
    from phasepack import phasecong
except ImportError:  # pragma: no cover - skimage is an optional dependency.
    print("phasepack not found, falling back to Laplacian for phase congruency.")
    phase_congruency = None


def _normalize(channel: np.ndarray) -> np.ndarray:
    """Normalize a 2D array to [0, 1] with numerical stability."""
    channel = channel.astype(np.float32)
    min_val = float(channel.min())
    max_val = float(channel.max())
    denom = max(max_val - min_val, 1e-8)
    channel = (channel - min_val) / denom
    return np.clip(channel, 0.0, 1.0)


# def _compute_sasad(gray: np.ndarray) -> np.ndarray:
#     """Approximate the SASAD descriptor via contrast enhancement + gradients."""
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(gray)
#     grad_x = cv2.Scharr(enhanced, cv2.CV_32F, 1, 0)
#     grad_y = cv2.Scharr(enhanced, cv2.CV_32F, 0, 1)
#     gradient = cv2.magnitude(grad_x, grad_y)
#     gradient = _normalize(gradient)

#     shadow = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=7)
#     shadow = _normalize(shadow)

#     descriptor = 0.65 * gradient + 0.35 * (1.0 - shadow)
#     return np.clip(descriptor, 0.0, 1.0)

# def _compute_sasad(gray: np.ndarray, num_iter: int = 15, delta_t: float = 0.14, q0_factor: float = 1.0) -> np.ndarray:
#     """
#     Real implementation of SASAD (Speckle Reducing Anisotropic Diffusion).
#     Solves the PDE: dI/dt = div( c(q) * grad(I) )
#     """
#     # 1. 初始类型转换 (float32, 0-1)
#     img = gray.astype(np.float32) / 255.0 if gray.dtype == np.uint8 else gray.astype(np.float32)
    
#     # 避免除零错误的小常数
#     eps = 1e-10

#     for t in range(num_iter):
#         # 2. 计算四个方向的梯度 (北, 南, 西, 东)
#         # N: (x, y-1) - (x, y)
#         dN = np.roll(img, -1, axis=0) - img
#         dS = np.roll(img, 1, axis=0) - img
#         dE = np.roll(img, -1, axis=1) - img
#         dW = np.roll(img, 1, axis=1) - img

#         # 3. 计算瞬时变异系数 q(x, y)
#         # 梯度模长近似
#         grad_mag = np.sqrt(dN**2 + dS**2 + dE**2 + dW**2 + eps)
#         # 局部均值 (简单的平滑作为基准)
#         local_mean = cv2.GaussianBlur(img, (5, 5), 1.0) + eps
        
#         # q = |Grad| / Mean
#         q = grad_mag / local_mean
        
#         # 4. 估计散斑基准值 q0 (取图像中相对平滑区域的q值)
#         # 简单的策略：取q的中位数或分位数作为基准噪声水平
#         q0 = np.median(q) * q0_factor
        
#         if q0 == 0: q0 = eps

#         # 5. 计算扩散系数 c(q)
#         # 公式: c(q) = 1 / (1 + (q^2 - q0^2) / (q0^2 * (1 + q0^2)))
#         # 下面的实现使用了简化的 SRAD 扩散函数形式，效果更稳定
#         denom = (q**2 - q0**2) / (q0**2 * (1 + q0**2) + eps)
#         c = 1.0 / (1.0 + np.maximum(denom, 0)) # 保证 denom >= 0

#         # 6. 更新图像 (PDE 离散化更新)
#         # div = cN*dN + cS*dS + cE*dE + cW*dW
#         # 注意：系数 c 应该是对应方向的。这里简化使用中心点 c，或者计算半点 c。
#         # 为保持高效，通常使用中心 c 近似
#         img += delta_t * (c * dN + c * dS + c * dE + c * dW)

#     return _normalize(img)

def _compute_sasad(gray: np.ndarray, num_iter: int = 5, delta_t: float = 0.14, q0_factor: float = 1.0) -> np.ndarray:
    """
    Real implementation of SASAD (Speckle Reducing Anisotropic Diffusion).
    Fixed: Uses np.pad instead of np.roll to prevent boundary wrap-around artifacts.
    """
    # 1. 初始类型转换
    img = gray.astype(np.float32) / 255.0 if gray.dtype == np.uint8 else gray.astype(np.float32)
    eps = 1e-10

    for t in range(num_iter):
        # 使用 symmetric (镜像) 或 edge (复制) 填充边界，防止梯度计算越界
        # 这里使用 'symmetric' 往往在扩散方程中效果更好
        img_pad = np.pad(img, ((1, 1), (1, 1)), mode='symmetric')

        # 中心区域 (对应原图 img)
        center = img_pad[1:-1, 1:-1]

        # 2. 计算四个方向的梯度 (利用切片操作，速度很快)
        # North: (y-1, x) - (y, x)  -> 对应 padding 后的 [0:-2, 1:-1] - center
        dN = img_pad[0:-2, 1:-1] - center
        # South: (y+1, x) - (y, x)  -> 对应 padding 后的 [2:, 1:-1] - center
        dS = img_pad[2:, 1:-1]   - center
        # East:  (y, x+1) - (y, x)  -> 对应 padding 后的 [1:-1, 2:] - center
        dE = img_pad[1:-1, 2:]   - center
        # West:  (y, x-1) - (y, x)  -> 对应 padding 后的 [1:-1, 0:-2] - center
        dW = img_pad[1:-1, 0:-2] - center

        # 3. 计算瞬时变异系数 q(x, y)
        grad_mag = np.sqrt(dN**2 + dS**2 + dE**2 + dW**2 + eps)
        local_mean = cv2.GaussianBlur(img, (5, 5), 1.0) + eps
        q = grad_mag / local_mean
        
        # 4. 估计散斑基准值 q0
        q0 = np.median(q) * q0_factor
        if q0 == 0: q0 = eps

        # 5. 计算扩散系数 c(q)
        denom = (q**2 - q0**2) / (q0**2 * (1 + q0**2) + eps)
        c = 1.0 / (1.0 + np.maximum(denom, 0))

        # 6. 更新图像
        img += delta_t * (c * dN + c * dS + c * dE + c * dW)

    return _normalize(img)


def _compute_phase_congruency(gray: np.ndarray) -> np.ndarray:
    """Generate a phase congruency response map using skimage when available."""
    gray_norm = gray.astype(np.float32) / 255.0
    if phasecong is not None:
        pc, *_ = phasecong(gray_norm)
        pc = np.nan_to_num(pc, nan=0.0, posinf=0.0, neginf=0.0)
    else:  # Fallback to Laplacian energy if skimage is missing.
        return print("phasepack not found, falling back to Laplacian for phase congruency.")
    return _normalize(pc)


def build_triplet_tensor(rgb_image: np.ndarray) -> np.ndarray:
    """
    Build the advanced three-channel tensor (Original + SASAD + PC).

    Args:
        rgb_image: RGB image (H, W, 3) in uint8.

    Returns:
        np.ndarray shaped (3, H, W) with float32 values in [0, 1].
    """
    rgb = np.clip(rgb_image, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    original = gray.astype(np.float32) / 255.0
    sasad = _compute_sasad(gray)
    pc = _compute_phase_congruency(gray)
    stacked = np.stack([original, sasad, pc], axis=0).astype(np.float32)
    return stacked


def _shadow_mask(height: int, width: int) -> np.ndarray:
    orientation = random.choice(["horizontal", "vertical", "diag"])
    start = 1.0
    end = random.uniform(0.15, 0.55)
    if orientation == "horizontal":
        gradient = np.linspace(start, end, height, dtype=np.float32).reshape(height, 1)
        mask = np.repeat(gradient, width, axis=1)
    elif orientation == "vertical":
        gradient = np.linspace(start, end, width, dtype=np.float32)
        mask = np.repeat(gradient.reshape(1, width), height, axis=0)
    else:  # diagonal shadow
        xv, yv = np.meshgrid(np.linspace(0, 1, width, dtype=np.float32),
                             np.linspace(0, 1, height, dtype=np.float32))
        diag = (xv + yv) / 2.0
        mask = start - (start - end) * diag
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=9)
    return np.clip(mask, 0.2, 1.0)


def _apply_occlusion(tensor: np.ndarray) -> np.ndarray:
    degraded = tensor.copy()
    _, height, width = degraded.shape
    occ_h = max(1, int(random.uniform(0.1, 0.25) * height))
    occ_w = max(1, int(random.uniform(0.1, 0.3) * width))
    top = random.randint(0, height - occ_h)
    left = random.randint(0, width - occ_w)
    attenuation = random.uniform(0.0, 0.4)
    degraded[:, top:top + occ_h, left:left + occ_w] *= attenuation
    return degraded


def _apply_shadow(tensor: np.ndarray) -> np.ndarray:
    mask = _shadow_mask(tensor.shape[1], tensor.shape[2])
    return tensor * mask[np.newaxis, :, :]


def _apply_deformation(tensor: np.ndarray) -> np.ndarray:
    degraded = []
    _, height, width = tensor.shape
    src = np.float32([[0, 0], [width - 1, 0], [0, height - 1]])
    max_offset = 0.08
    offsets = np.array([
        [random.uniform(-max_offset, max_offset) * width, random.uniform(-max_offset, max_offset) * height]
        for _ in range(3)
    ], dtype=np.float32)
    dst = src + offsets
    matrix = cv2.getAffineTransform(src, dst)
    for channel in tensor:
        warped = cv2.warpAffine(
            channel,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            # borderMode=cv2.BORDER_REFLECT_101
        )
        degraded.append(warped)
    return np.stack(degraded, axis=0)


def degrade_triplet(tensor: np.ndarray,
                    shadow_prob: float = 0.65,
                    occlusion_prob: float = 0.55,
                    deformation_prob: float = 0.4) -> np.ndarray:
    """
    Apply degradations (shadowing, occlusion, deformation) to simulate adverse inputs.

    Args:
        tensor: (3, H, W) float32 tensor produced by ``build_triplet_tensor``.

    Returns:
        Degraded tensor with the same shape and dtype.
    """
    degraded = tensor.copy()
    if random.random() < shadow_prob:
        degraded = _apply_shadow(degraded)
    if random.random() < occlusion_prob:
        degraded = _apply_occlusion(degraded)
    if random.random() < deformation_prob:
        degraded = _apply_deformation(degraded)
    return np.clip(degraded, 0.0, 1.0).astype(np.float32)
