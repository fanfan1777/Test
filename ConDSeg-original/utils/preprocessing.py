import random

import cv2
import numpy as np

try:
    from skimage.feature import phase_congruency
except ImportError:  # pragma: no cover - skimage is an optional dependency.
    phase_congruency = None


def _normalize(channel: np.ndarray) -> np.ndarray:
    """Normalize a 2D array to [0, 1] with numerical stability."""
    channel = channel.astype(np.float32)
    min_val = float(channel.min())
    max_val = float(channel.max())
    denom = max(max_val - min_val, 1e-8)
    channel = (channel - min_val) / denom
    return np.clip(channel, 0.0, 1.0)


def _compute_sasad(gray: np.ndarray) -> np.ndarray:
    """Approximate the SASAD descriptor via contrast enhancement + gradients."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    grad_x = cv2.Scharr(enhanced, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(enhanced, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(grad_x, grad_y)
    gradient = _normalize(gradient)

    shadow = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=7)
    shadow = _normalize(shadow)

    descriptor = 0.65 * gradient + 0.35 * (1.0 - shadow)
    return np.clip(descriptor, 0.0, 1.0)


def _compute_phase_congruency(gray: np.ndarray) -> np.ndarray:
    """Generate a phase congruency response map using skimage when available."""
    gray_norm = gray.astype(np.float32) / 255.0
    if phase_congruency is not None:
        pc, *_ = phase_congruency(gray_norm)
        pc = np.nan_to_num(pc, nan=0.0, posinf=0.0, neginf=0.0)
    else:  # Fallback to Laplacian energy if skimage is missing.
        pc = np.abs(cv2.Laplacian(gray_norm, cv2.CV_32F, ksize=3))
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
            borderMode=cv2.BORDER_REFLECT_101
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
