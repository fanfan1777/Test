import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from physics_utils import build_triplet_tensor

# 配置路径
DATASET_ROOT = "/workspace/data/TN3K"  # 修改为你的数据集根目录
PROCESS_CONFIG = [
    ("trainval-image", "image"),
    ("trainval-mask",  "mask"),
    # ("test-image",     "image"),
    # ("test-mask",      "mask")
]


def process_single_file(task_info):
    """
    单个文件处理函数，用于多进程调用
    """
    input_path, output_path, mode = task_info

    # 如果已经存在，可以选择跳过
    # if os.path.exists(output_path): return

    try:
        if mode == "image":
            # --- 图像处理模式 ---
            img = cv2.imread(input_path)
            if img is None: raise ValueError("Image read failed")

            # 计算三通道物理特征 (Raw, SASAD, PC)
            # 返回 float32, range [0, 1], shape (H, W, 3)
            data = build_triplet_tensor(img)

        elif mode == "mask":
            # --- 掩码处理模式 ---
            # 读取为单通道灰度图
            mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: raise ValueError("Mask read failed")

            # 保持 uint8 格式 (0-255) 以节省空间，shape (H, W)
            # 训练时读取后再除以 255.0
            data = mask

        else:
            return

        # 保存为 .npy
        np.save(output_path, data)

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")


def main():
    print(f"开始全量预处理 (Images & Masks)... CPU核心数: {cpu_count()}")

    tasks = []

    for folder, mode in PROCESS_CONFIG:
        input_dir = os.path.join(DATASET_ROOT, folder)
        output_dir = os.path.join(DATASET_ROOT, "preprocessed", folder)

        # 检查输入目录是否存在
        if not os.path.exists(input_dir):
            print(f"跳过不存在的目录: {input_dir}")
            continue

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")

        # 获取所有图片/掩码文件
        extensions = ['*.jpg', '*.png', '*.jpeg', '*.bmp', '*.tif']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(input_dir, ext)))

        print(f"[{mode.upper()}] 文件夹 {folder} 找到 {len(files)} 个文件")

        # 构建任务列表
        for file_path in files:
            file_name = os.path.basename(file_path)
            # 统一修改扩展名为 .npy
            file_name_npy = os.path.splitext(file_name)[0] + ".npy"
            output_path = os.path.join(output_dir, file_name_npy)

            # 任务元组: (输入路径, 输出路径, 处理模式)
            tasks.append((file_path, output_path, mode))

    # 多进程处理
    processes = max(1, cpu_count() - 2)

    with Pool(processes=processes) as pool:
        list(tqdm(pool.imap(process_single_file, tasks), total=len(tasks), desc="Converting to .npy"))

    print("所有图像和掩码已完成 .npy 转换！")

if __name__ == '__main__':
    main()