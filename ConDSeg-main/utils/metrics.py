import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft  # 新增: 用于频域一致性计算
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import convolve

""" --- Existing Helper Functions --- """


def Object(pred, gt):
    x = np.mean(pred[gt == 1])
    sigma_x = np.std(pred[gt == 1])
    score = 2.0 * x / (x ** 2 + 1 + sigma_x + np.finfo(np.float64).eps)
    return score


def S_Object(pred, gt):
    pred_fg = pred.copy()
    pred_fg[gt != 1] = 0.0
    O_fg = Object(pred_fg, gt)

    pred_bg = (1 - pred.copy())
    pred_bg[gt == 1] = 0.0
    O_bg = Object(pred_bg, 1 - gt)

    u = np.mean(gt)
    Q = u * O_fg + (1 - u) * O_bg
    return Q


def centroid(gt):
    if np.sum(gt) == 0:
        return gt.shape[0] // 2, gt.shape[1] // 2
    else:
        x, y = np.where(gt == 1)
        return int(np.mean(x).round()), int(np.mean(y).round())


def divide(gt, x, y):
    LT = gt[:x, :y]
    RT = gt[x:, :y]
    LB = gt[:x, y:]
    RB = gt[x:, y:]

    w1 = LT.size / gt.size
    w2 = RT.size / gt.size
    w3 = LB.size / gt.size
    w4 = RB.size / gt.size

    return LT, RT, LB, RB, w1, w2, w3, w4


def ssim(pred, gt):
    x = np.mean(pred)
    y = np.mean(gt)
    N = pred.size

    sigma_x2 = np.sum((pred - x) ** 2 / (N - 1 + np.finfo(np.float64).eps))
    sigma_y2 = np.sum((gt - y) ** 2 / (N - 1 + np.finfo(np.float64).eps))
    sigma_xy = np.sum((pred - x) * (gt - y) / (N - 1 + np.finfo(np.float64).eps))

    alpha = 4 * x * y * sigma_xy
    beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        Q = alpha / (beta + np.finfo(np.float64).eps)
    elif alpha == 0 and beta == 0:
        Q = 1
    else:
        Q = 0
    return Q


def S_Region(pred, gt):
    x, y = centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = divide(gt, x, y)
    pred1, pred2, pred3, pred4, _, _, _, _ = divide(pred, x, y)

    Q1 = ssim(pred1, gt1)
    Q2 = ssim(pred2, gt2)
    Q3 = ssim(pred3, gt3)
    Q4 = ssim(pred4, gt4)

    Q = Q1 * w1 + Q2 * w2 + Q3 * w3 + Q4 * w4
    return Q


def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def AlignmentTerm(pred, gt):
    mu_pred = np.mean(pred)
    mu_gt = np.mean(gt)

    align_pred = pred - mu_pred
    align_gt = gt - mu_gt

    align_mat = 2 * (align_gt * align_pred) / (align_gt ** 2 + align_pred ** 2 + np.finfo(np.float64).eps)
    return align_mat


def EnhancedAlighmentTerm(align_mat):
    enhanced = ((align_mat + 1) ** 2) / 4
    return enhanced


""" --- Loss Functions --- """


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        inputs = torch.clamp(inputs, 1e-7, 1.0 - 1e-7)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        return Dice_BCE


class MultiClassBCE(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        loss = 0
        for i in range(inputs.shape[1]):
            yp = inputs[:, i]
            yt = targets[:, i]
            BCE = F.binary_cross_entropy(yp, yt, reduction='mean')
            loss += BCE
        return loss


# ==============================================================================
# NEW: Ultra-ConDSeg Consistency Loss (Scheme 3 + Recommended Scheme B)
# ==============================================================================
class UltraSoundConsistencyLoss(nn.Module):
    """
    包含了针对超声图像优化的三种一致性损失：
    1. Soft KL Divergence: 概率分布对齐
    2. Gradient Consistency: 几何结构/边缘对齐 (带平滑处理)
    3. Frequency Domain Consistency: 幅度谱与相位谱对齐
    """

    def __init__(self, weight_skl=10.0, weight_grad=1.0, weight_freq=0.1):
        super(UltraSoundConsistencyLoss, self).__init__()
        self.weight_skl = weight_skl
        self.weight_grad = weight_grad
        self.weight_freq = weight_freq

        # 定义 Sobel 核用于梯度计算
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    def soft_kl_loss(self, p1, p2,weight_map = None):
        """
        Symmetric Soft KL Divergence.
        公式: P1 * log(P1/P2) + P2 * log(P2/P1)
        """
        eps = 1e-8
        # 确保数值稳定性
        p1 = torch.clamp(p1, eps, 1.0 - eps)
        p2 = torch.clamp(p2, eps, 1.0 - eps)

        loss = (p1 * (torch.log(p1) - torch.log(p2))) + (p2 * (torch.log(p2) - torch.log(p1)))
        if weight_map is not None:
            loss = loss * weight_map

        return loss.mean()

    def gradient_loss(self, p1, p2):
        """
        Gradient Consistency with Smoothing.
        计算 Sobel 梯度差异，预先使用 AvgPool 平滑以抑制超声散斑噪声带来的梯度震荡。
        """
        # 1. 预平滑 (Numerical Stability Strategy)
        p1_smooth = F.avg_pool2d(p1, kernel_size=3, stride=1, padding=1)
        p2_smooth = F.avg_pool2d(p2, kernel_size=3, stride=1, padding=1)

        # 确保 kernel 在正确的 device 上
        if self.sobel_x.device != p1.device:
            self.sobel_x = self.sobel_x.to(p1.device)
            self.sobel_y = self.sobel_y.to(p1.device)

        # 2. 计算梯度
        g1_x = F.conv2d(p1_smooth, self.sobel_x, padding=1)
        g1_y = F.conv2d(p1_smooth, self.sobel_y, padding=1)

        g2_x = F.conv2d(p2_smooth, self.sobel_x, padding=1)
        g2_y = F.conv2d(p2_smooth, self.sobel_y, padding=1)

        # 3. L2 范数差异
        loss_x = F.mse_loss(g1_x, g2_x)
        loss_y = F.mse_loss(g1_y, g2_y)

        return loss_x + loss_y

    def frequency_loss(self, p1, p2):
        """
        Frequency Domain Consistency (Scheme B).
        分别约束幅度谱(Amplitude)和相位谱(Phase)。
        """
        # 1. FFT 变换 (Real FFT 2D)
        fft1 = torch.fft.rfft2(p1, norm='ortho')
        fft2 = torch.fft.rfft2(p2, norm='ortho')

        # 2. 提取幅度和相位
        amp1 = torch.abs(fft1)
        amp2 = torch.abs(fft2)

        phase1 = torch.angle(fft1)
        phase2 = torch.angle(fft2)

        # 3. 计算损失
        # 幅度谱损失：保持整体风格和平滑度
        # loss_amp = F.mse_loss(amp1, amp2)
        loss_amp = F.mse_loss(torch.log(amp1 + 1.0), torch.log(amp2 + 1.0))
        # 相位谱损失：保持结构和几何对齐 (关键)
        loss_phase = F.mse_loss(phase1, phase2)

        return loss_amp + loss_phase

    def forward(self, pred1, pred2,weight_map = None):
        """
        pred1, pred2: 预测的概率图 (Batch, Channel, H, W)，值域 [0, 1]
        weight_map: (Batch, 1, H, W) 像素级权重图，值域 [0, 1]
        """
        loss_skl = self.soft_kl_loss(pred1, pred2,weight_map)
        loss_grad = self.gradient_loss(pred1, pred2)
        loss_freq = self.frequency_loss(pred1, pred2)

        total_loss = (self.weight_skl * loss_skl +
                      self.weight_grad * loss_grad +
                      self.weight_freq * loss_freq)

        return total_loss


""" --- Metrics --- """


def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)


def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)


def F2(y_true, y_pred, beta=2):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (1 + beta ** 2.) * (p * r) / float(beta ** 2 * p + r + 1e-15)


def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def mae(y_true, y_pred):
    sum = 0
    for i in range(len(y_true)):
        sum += abs(y_true[i] - y_pred[i])
    return sum / len(y_true)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

#TODO 新增
class SDFConsistencyLoss(nn.Module):
    """
    SDF Consistency Loss (Scheme A)
    计算两个增强视图下 SDF 预测图的均方误差 (MSE)。
    SDF 隐式定义了全局几何拓扑，强制 SDF 一致能有效防止拓扑断裂。
    """
    def __init__(self, weight=1.0):
        super(SDFConsistencyLoss, self).__init__()
        self.weight = weight

    def forward(self, sdf1, sdf2,weight_map = None):
        """
        Args:
            sdf1: 原始视图的 SDF 预测 (Batch, 1, H, W)
            sdf2: 增强视图的 SDF 预测 (Batch, 1, H, W)
            weight_map: (Batch, 1, H, W) 像素级权重图。
            对于 SDF，传入的权重应该是 (1 - Uncertainty)。
        """
        loss = F.mse_loss(sdf1, sdf2, reduction='none')

        if weight_map is not None:
            loss = loss * weight_map


        return self.weight * loss.mean()
