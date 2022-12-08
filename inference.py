import gc
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader  # noqa

from utils import util
from utils import dataset
from models.unet import UNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_type', type=str, choices=["Temporal", "Spatial"], default='Temporal')
    parser.add_argument('--cities', type=list, choices=['ANTWERP', 'BANGKOK', 'BARCELONA', "MOSCOW"], default=['BANGKOK'])
    parser.add_argument('--train_year', type=list, choices=[2019, 2020], default=[2020])
    parser.add_argument('--val_year', type=list, choices=[2019, 2020], default=[2020])
    parser.add_argument('--model', type=str, choices=["UNET"], default="UNET")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_mask', type=bool, default=False)
    parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--seed', type=int, default=14563)
    return parser.parse_args()


def inference(args, path, writer, experiment_name):

    _, valid_files, static_map = util.get_files(args)

    test_dataset = dataset.TrainDataset(valid_files[0:15], static_map)  # using 15/42 files
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=dataset.collate_fn,  # noqa
                                sampler=dataset.val_local_sampler(test_dataset), pin_memory=True)  # noqa

    model = UNet()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda().eval()
    criterian = torch.nn.MSELoss()

    test_loss = 0.0
    running_loss = 0.0
    dataset_size = 0
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing")
        for idx, data in pbar:
            inputs = data[0].to(args.device, dtype=torch.float)
            targets = data[1].to(args.device, dtype=torch.float)

            prediction = model(inputs)
            pred = prediction[:, :, 1:, 6:-6]
            loss = criterian(pred, targets)

            dataset_size += inputs.shape[0]
            running_loss += loss.item() * inputs.shape[0]
            test_loss = running_loss / dataset_size
            pbar.set_postfix(test_loss=f"{test_loss:0.4f}")

            test_targets.append(targets.cpu().numpy())
            test_predictions.append(pred.cpu().numpy())

            torch.cuda.empty_cache()
            gc.collect()

    test_predictions = np.concatenate(test_predictions, axis=0).astype(np.float32)
    test_predictions = test_predictions.reshape(-1, 6, 8, 495, 436)
    test_predictions = np.moveaxis(test_predictions, source=2, destination=4)  # (n, 6, 495, 436, 8)

    test_targets = np.concatenate(test_targets, axis=0).astype(np.float32)
    test_targets = test_targets.reshape(-1, 6, 8, 495, 436)
    test_targets = np.moveaxis(test_targets, source=2, destination=4)  # (n, 6, 495, 436, 8)

    util.plot(test_predictions, test_targets, experiment_name)


if __name__ == "__main__":

    start = time.time()

    args = parse_args()
    util.seed_everything(args.seed)

    experiment_name = "Training_2019_Validation_2020_ANTWERP_UNET_unet512filter_L1_dropout"

    writer = SummaryWriter(f"logs/{experiment_name}")

    path_to_best = f"checkpoints/{experiment_name}/best_model.pth"
    path_to_last = f"checkpoints/{experiment_name}/last_model.pth"

    inference(args, path_to_best, writer, experiment_name)

    print("Inference_time: ", round((time.time() - start)/60, 2), "(min)")
