import os
import re
import gc
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from utils.dataset import GenerateTrainFiles

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


def get_files(args, is_test=False):
    training_files = GenerateTrainFiles(args, is_test)
    static_map = training_files.files[2]
    if is_test:
        test_data = training_files.dynamic_train_files
        return test_data, static_map
    train_data, valid_data = training_files.dynamic_train_files[0], training_files.dynamic_train_files[1]
    return train_data, valid_data, static_map


def evaluate(args, model, dataloader, static_map, criterian, epoch):
    model.eval()
    val_epoch_loss = 0.0
    running_loss = 0.0
    dataset_size = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Validation {epoch+1}/{args.num_epochs}")
        for idx, data in pbar:
            inputs = data[0].to(args.device, dtype=torch.float)
            targets = data[1].to(args.device, dtype=torch.float)

            prediction = model(inputs)
            if args.use_mask:
                mask_city = create_static_mask(args, static_map)
                pred = prediction[:, :, 1:, 6:-6] * mask_city
            else:
                pred = prediction[:, :, 1:, 6:-6]
            loss = criterian(pred, targets)

            dataset_size += inputs.shape[0]
            running_loss += loss.item() * inputs.shape[0]
            val_epoch_loss = running_loss / dataset_size
            pbar.set_postfix(eval_loss=f"{val_epoch_loss:0.5f}")

            torch.cuda.empty_cache()
            gc.collect()
    model.train()
    return val_epoch_loss


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def create_static_mask(args, static_data: dict):
    if args.use_mask:
        static_masks = {}
        for city, static_map in static_data.items():
            assert static_map.shape == (9, 495, 436)
            mask = np.where(static_map[0] > 0, 1, 0)
            static_masks[city] = mask  # 495, 436
        mask_city = torch.tensor(static_masks[args.cities[0]]).float().cuda()
        mask_city = mask_city[None, None, :, :]
        return mask_city
    return None


def log_to_tensorboard(writer, model, train_loss, eval_loss, current_lr, epoch, save_histogram=False):
    writer.add_scalar("Training_loss/epochs", train_loss, epoch)
    writer.add_scalar("validation_loss/epochs", eval_loss, epoch)
    writer.add_scalar("learning_rates/epochs", current_lr, epoch)
    if save_histogram:
        for name, weight in model.named_parameters():
            writer.add_histogram(name, weight, epoch)
            writer.add_histogram(f'{name}.grad', weight.grad, epoch)


def get_scheduler(scheduler, optimizer):
    if scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=100, threshold=0.001, min_lr=1e-6, verbose=True)
    elif scheduler == "StepLR":
        scheduler = StepLR(optimizer, step_size=2, gamma=0.7)
    elif scheduler is None:
        return None
    return scheduler


def plot(test_predictions, test_targets, experiment, title):

    fig, ax = plt.subplots(2, 6, figsize=(18, 8))
    ax = ax.ravel()

    itot = {0: '5 (min)',  1: '10 (min)', 2: '15 (min)', 3: '30 (min)', 4: '45 (min)', 5: '60 (min)'}

    idx_to_plot = np.random.randint(0, test_predictions.shape[0], 1)  # selecting one tensor to plot (n, 6, 495, 436, 8)
    predicted_img = test_predictions[idx_to_plot][0]  # (6,495,436,8)
    ground_truth = test_targets[idx_to_plot][0]  # (6,495,436,8)

    for i in range(6):
        pred_img = predicted_img[i]  # (495,436,8)
        target_img = ground_truth[i]  # (495,436,8)
        sum_pred = np.zeros((495, 436))
        sum_target = np.zeros((495, 436))
        for ch in range(8):  # 8 channel
            channel_pred = pred_img[:, :, ch]  # (495,436)
            channel_target = target_img[:, :, ch]  # (495,436)
            sum_pred += channel_pred
            sum_target += channel_target
        ax[i].imshow(sum_pred)
        ax[i].set_title(f"Prediction for {itot[i]}")
        ax[i+6].imshow(sum_target)
        ax[i+6].set_title(f"Ground Truth for {itot[i]}")
    plt.tight_layout(pad=0.3)
    fig.suptitle(f'Prediction vs Ground Truth for {title}')
    plt.savefig(f"./images/{experiment}+{title}_test")


def inputs_sanity_checks(files):
    for idx, file in enumerate(files):
        city = re.search(r"([A-Z]+)", str(file)).group(1)
        date = re.search(r"([0-9]+-[0-9]+-[0-9]+)", str(file)).group(1)
        days = datetime.strptime(date, "%Y-%m-%d").weekday()
        months = datetime.strptime(date, "%Y-%m-%d").strftime("%b")
        year = datetime.strptime(date, "%Y-%m-%d").year
        print(f"{city=}, {days=}, {months=}, {year=}")
        if (idx + 1) % 28 == 0:
            print("*"*20)
