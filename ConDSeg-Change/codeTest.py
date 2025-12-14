import sys
import os
import cv2
import numpy as np
import torch

# 确保可以导入 utils 模块
sys.path.append(os.getcwd())

from utils.preprocessing import build_triplet_tensor, degrade_triplet

def test_input_logic():
    print("=== 开始测试数据输入逻辑 ===")
    
    # 1. 模拟一张随机的 RGB 图像 (256x256)
    # 注意：真实数据读取是 OpenCV BGR -> RGB，这里直接模拟 RGB
    H, W = 256, 256
    fake_image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    
    print(f"1. 生成模拟图像，形状: {fake_image.shape}")

    # 2. 生成 Input 1 (Primary)
    # 对应 utils/run_engine.py 中的 primary = build_triplet_tensor(image)
    primary = build_triplet_tensor(fake_image)
    
    # 验证 Input 1
    print(f"2. 生成 Input 1 (Primary Tensor)")
    print(f"   - 形状 (C, H, W): {primary.shape}")
    print(f"   - 数据类型: {primary.dtype}")
    print(f"   - 数值范围: [{primary.min():.4f}, {primary.max():.4f}]")
    
    if primary.shape[0] != 3:
        print("   [错误] Input 1 通道数不是 3！")
    else:
        print("   [正确] Input 1 是三通道张量 (Original + SASAD + PC)。")

    # 3. 生成 Input 2 (Degraded)
    # 对应 utils/run_engine.py 中的 degraded = degrade_triplet(primary)
    # 强制让破坏发生概率为 100% 以便测试 (通过修改函数参数默认值或多次运行，这里假设默认参数足够触发)
    # 注意：utils.preprocessing.degrade_triplet 内部有概率判断，
    # 为了测试差异，我们这里手动调用内部函数或者假设概率触发。
    # 这里的 degrade_triplet 默认有较高概率触发 shadow/occlusion/deformation
    degraded = degrade_triplet(primary, shadow_prob=0.9, occlusion_prob=0.9, deformation_prob=0.9)

    print(f"3. 生成 Input 2 (Degraded Tensor)")
    print(f"   - 形状 (C, H, W): {degraded.shape}")
    
    # 4. 验证 Input 1 和 Input 2 的关系
    diff = np.abs(primary - degraded)
    mae = np.mean(diff)
    
    print(f"4. 对比验证")
    if primary.shape != degraded.shape:
        print("   [错误] Input 1 和 Input 2 形状不一致！")
        return

    print(f"   - Input 1 与 Input 2 的平均绝对误差 (MAE): {mae:.6f}")
    
    if mae > 0:
        print("   [正确] Input 2 与 Input 1 内容不同，说明已经叠加了“破坏”。")
        print("   [正确] 逻辑符合：Input 2 是在 Input 1 (三通道张量) 基础上进行的破坏。")
    else:
        print("   [警告] Input 2 与 Input 1 完全相同。可能是概率未触发破坏，或者是代码逻辑有误。")
        print("   (如果是概率问题，请尝试重新运行脚本)")

    # 5. 验证是否复制了原始图像
    # 如果代码写错，Input 2 可能是原始图像的破坏版而不是 Primary 的破坏版
    # 我们可以检查 degraded 的通道 1 和 2 (SASAD 和 PC) 是否有值
    # 如果它是原始图像，通道 1 和 2 应该是 G 和 B 通道，但在 Primary 中它们是特殊特征
    sasad_channel = degraded[1, :, :]
    pc_channel = degraded[2, :, :]
    
    if np.max(sasad_channel) <= 1.0 and np.max(pc_channel) <= 1.0:
        print("   [正确] Input 2 的数值范围在 [0, 1] 之间，符合归一化后的特征张量。")
    else:
        print("   [错误] Input 2 的数值超过 1.0，可能错误地使用了原始 uint8 图像。")

if __name__ == "__main__":
    try:
        test_input_logic()
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()