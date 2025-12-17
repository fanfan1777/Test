import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import torch
import argparse
from torch.utils.data import DataLoader

from utils.utils import print_and_save, shuffling, epoch_time
from network.model_stage1 import ConDSegStage1
from utils.metrics import DiceBCELoss

#目前已经修改了有resize，和两个数据增强的部分，其他并未进行修改
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from utils.run_engine_stage1 import load_data, train, evaluate, DATASET
def my_seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='Select GPU card (e.g., 0 or 1)')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    base_dir = "/workspace/data"
    dataset_name = 'TN3K'
    preprocessed_name = 'preprocessed'
    val_name=None

    seed = 0

    my_seeding(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU ID (Physical): {args.gpu}")
    print(f"Using Device (Logical): {device}")

    image_size = 256
    size = (image_size, image_size)
    batch_size = 8
    num_epochs = 200
    lr = 1e-4
    early_stopping_patience = 100
    val_split = 0.15

    resume_path = '/workspace/ConDSeg-main/run_files/TN3K/stage1_TN3K_None_lr0.0001_20251216-141638/checkpoint.pth'


    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"stage1_{dataset_name}_{val_name}_lr{lr}_{current_time}"


    data_path = os.path.join(base_dir, dataset_name,preprocessed_name)
    save_dir = os.path.join("run_files", dataset_name, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_log_path = os.path.join(save_dir, "train_log.txt")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    train_log = open(train_log_path, "w")
    train_log.write("\n")
    train_log.close()

    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    hyperparameters_str = f"Image Size: {image_size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    hyperparameters_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    hyperparameters_str += f"Seed: {seed}\n"
    hyperparameters_str += f"Validation Split: {val_split}\n"
    print_and_save(train_log_path, hyperparameters_str)

    """ Data augmentation: Transforms """

    transform = A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ElasticTransform(alpha=50, sigma=120 * 0.06, alpha_affine=20, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=12, max_width=12, p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5)
    ])

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y) = load_data(
        data_path,
        val_name,
        val_split=val_split,
        seed=seed
    )
    train_x, train_y = shuffling(train_x, train_y)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset and loader """
    train_dataset = DATASET(train_x,  train_y, (image_size, image_size), transform=transform)
    valid_dataset = DATASET(valid_x,  valid_y, (image_size, image_size), transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ Model """
    # device = torch.device('cuda')
    model = ConDSegStage1()

    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, verbose=True)
    loss_fn=DiceBCELoss()
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)


    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    data_str = f"Number of parameters: {num_params/1000000}M\n"
    print_and_save(train_log_path, data_str)


    """ Training the model """
    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_metrics[0] > best_valid_metrics:
            data_str = f"Valid mIoU improved from {best_valid_metrics:2.4f} to {valid_metrics[0]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[0]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0


        elif valid_metrics[0] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - mIoU: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - mIoU: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break