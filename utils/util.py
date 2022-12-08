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


def get_files(args):
    """
    This function returns the generated files for training and validation

    Returns:
        train_files, valid_files, static_map
    """
    generated_files = GenerateTrainFiles(args)
    static_map = generated_files.files[2]
    train_files = generated_files.train_files
    valid_files = generated_files.valid_files
    return train_files, valid_files, static_map


def evaluate(args, model, dataloader, static_map, criterian, epoch, writer, experiment):
    """
    This function evaluates the performance of the model on the validation files.

    Args:
        args: args
        model : model
        dataloader : validation data loader
        static_map : static data
        criterian : Loss function
        epoch: current epoch
        writer: tensorboard summary writer
        experiment: name of the current experimnet

    Returns:
        validation loss for epoch
    """
    model.eval()
    val_epoch_loss = 0.0
    running_loss = 0.0
    dataset_size = 0
    test_predictions = []
    test_targets = []
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

            writer.add_scalar("validation_loss/minibatches", val_epoch_loss, idx + (epoch*len(dataloader)))

            if (epoch+1) == args.num_epochs and (idx > 780):
                test_targets.append(targets.cpu().numpy())
                test_predictions.append(pred.cpu().numpy())

            torch.cuda.empty_cache()
            gc.collect()

    if (epoch+1) == args.num_epochs:
        plot(test_predictions, test_targets, experiment)

    model.train()
    return val_epoch_loss


def seed_everything(seed: int):
    """
    This function seeds every thing for reproducibility

    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def create_static_mask(args, static_data: dict):
    """
    This function creates masks for cities using the static data.

    Args:
        args: determines the training city
        static_data (dict): containing static data for cities ex:{"ANTWERP":(9, 495, 436)}

    Returns:
        torch tensor (495, 436)
    """
    static_masks = {}
    for city, static_map in static_data.items():
        assert static_map.shape == (9, 495, 436)
        mask = np.where(static_map[0] > 0, 1, 0)
        static_masks[city] = mask  # 495, 436
    mask_city = torch.tensor(static_masks[args.cities[0]]).float().cuda()
    return mask_city


def log_to_tensorboard(writer, model, train_loss, eval_loss, current_lr, epoch, save_histogram=False):
    """
    This function saves the performance measurements of the model for each epoch during training.
    The user can observe them using tensorboard.

    Args:
        writer: tensorboard Summary Writer
        model: model
        train_loss: Training loss for each epoch
        eval_loss: Validation loss for each epoch
        current_lr: epoch's Learning rate based on the scheduler
        epoch: current epoch
        save_histogram (bool, optional): histogram of weights, grads and biases for each layer. Defaults to False.
    """
    writer.add_scalar("Training_loss/epochs", train_loss, epoch+1)
    writer.add_scalar("validation_loss/epochs", eval_loss, epoch+1)
    writer.add_scalar("learning_rates/epochs", current_lr, epoch+1)
    if save_histogram:
        for name, weight in model.named_parameters():
            writer.add_histogram(name, weight, epoch+1)
            writer.add_histogram(f'{name}.grad', weight.grad, epoch+1)


def get_scheduler(scheduler: str, optimizer):
    """
    This function returns the learning rate scheduler

    Args:
        scheduler
        optimizer

    Returns:
        scheduler
    """
    if scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=100, threshold=0.001, min_lr=1e-6, verbose=True)
    elif scheduler == "StepLR":
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    elif scheduler is None:
        return None
    return scheduler


def plot(test_predictions, test_targets, experiment):  # check for inputs dim for inference
    """This function plots prediction vs ground Truth for a random index of validation dataset.
        plots will be saved in the images directories.

    Args:
        test_predictions : predictions
        test_targets : ground Truth
        experiment : name of the current experiment
    """
    # if inference !!!!!!!!!!!!!!!
    test_predictions = np.concatenate(test_predictions, axis=0).astype(np.float32)
    test_predictions = test_predictions.reshape(-1, 6, 8, 495, 436)
    test_predictions = np.moveaxis(test_predictions, source=2, destination=4)  # (n, 6, 495, 436, 8)

    test_targets = np.concatenate(test_targets, axis=0).astype(np.float32)
    test_targets = test_targets.reshape(-1, 6, 8, 495, 436)
    test_targets = np.moveaxis(test_targets, source=2, destination=4)  # (n, 6, 495, 436, 8)

    fig, ax = plt.subplots(4, 4, figsize=(18, 18))  # 2, 6, 8
    ax = ax.ravel()

    itot = {0: '5 (min)',  1: '10 (min)', 2: '15 (min)', 3: '30 (min)', 4: '45 (min)', 5: '60 (min)'}

    idx_to_plot = np.random.randint(0, test_predictions.shape[0], 1)  # selecting one tensor to plot (n, 6, 495, 436, 8)
    predicted_img = test_predictions[idx_to_plot][0]  # (6,495,436,8)
    ground_truth = test_targets[idx_to_plot][0]  # (6,495,436,8)

    for i in range(6):
        pred_img = predicted_img[i+2, :, :, :]  # (495,436,8)
        target_img = ground_truth[i+2, :, :, :]  # (495,436,8)
        sum_pred = np.zeros((495, 436))
        sum_target = np.zeros((495, 436))
        for ch in range(8):  # 8 channel
            channel_pred = pred_img[:, :, ch]  # (495,436)
            channel_target = target_img[:, :, ch]  # (495,436)
            sum_pred += channel_pred
            sum_target += channel_target
        ax[ch].imshow(channel_pred)
        ax[ch].set_title(f"Prediction for {itot[i]}, {ch=}")
        ax[ch+8].imshow(channel_target)
        ax[ch+8].set_title(f"Ground Truth for {itot[i]}, {ch=}")
    plt.tight_layout(pad=0.5)
    fig.suptitle('Prediction vs Ground Truth for tempral domain shift (2020)')
    plt.savefig(f"./images/{experiment}")
    # plt.show()


def inputs_sanity_checks(files):
    """
    it can be used to check if the generated files for training and validation are as they expected to be.
    It prints out the information such as (city, day, month, year) for each single file.

    Args:
        files : generated files for training and validation
    """
    for idx, file in enumerate(files):
        city = re.search(r"([A-Z]+)", str(file)).group(1)
        date = re.search(r"([0-9]+-[0-9]+-[0-9]+)", str(file)).group(1)
        days = datetime.strptime(date, "%Y-%m-%d").weekday()
        months = datetime.strptime(date, "%Y-%m-%d").strftime("%b")
        year = datetime.strptime(date, "%Y-%m-%d").year
        print(f"{city=}, {days=}, {months=}, {year=}")


def make_dir():
    """
    This function checks whether necessary directories exist or not. if not,
    it creates three directores called checkpoints,logs and images in the main directory.

    """
    if not os.path.isdir("checkpoints"):
        os.makedirs("checkpoints")
        print("A new directory 'checkpoints' created for saving models\n")
    if not os.path.isdir("logs"):
        os.makedirs("logs")
        print("A new directory 'logs' created for tensorboard\n")
    if not os.path.isdir("images"):
        os.makedirs("images")
        print("A new directory 'images' created for saving images\n")
