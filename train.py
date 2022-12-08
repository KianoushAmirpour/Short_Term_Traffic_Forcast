import os
import gc
import time
import argparse
from tqdm import tqdm

import torch
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils import util
from utils import dataset
from models.unet import UNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cities', type=list, choices=['ANTWERP', 'BANGKOK', 'BARCELONA', "MOSCOW"], default=['ANTWERP'])
    parser.add_argument('--train_year', type=list, choices=[2019, 2020], default=[2019])
    parser.add_argument('--val_year', type=list, choices=[2019, 2020], default=[2020])
    parser.add_argument('--model', type=str, choices=["UNET"], default="UNET")
    parser.add_argument('--scheduler', type=str, default="StepLR")
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=6)
    parser.add_argument('--L1_regularization', type=bool, default=False)
    parser.add_argument('--wd', type=float, default=2e-4)
    parser.add_argument('--num_file_train', type=int, default=14)
    parser.add_argument('--accumulation_step', type=int, default=8)
    parser.add_argument('--use_mask', type=bool, default=False)
    parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--seed', type=int, default=14563)
    return parser.parse_args()


def train(args, writer, experiment_name):

    train_files, valid_files, static_map = util.get_files(args)

    valid_dataset = dataset.TrainDataset(valid_files, static_map)

    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, collate_fn=dataset.collate_fn,
                                  sampler=dataset.val_local_sampler(valid_dataset), pin_memory=True)

    if args.model == "UNET":
        model = UNet()
        model.to(args.device)
    criterian = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.wd)
    scheduler = util.get_scheduler(args.scheduler, optimizer)
    scaler = amp.GradScaler()

    best_val_loss = 1e10
    dataset_size = 0
    global_step = 0

    for epoch in range(args.num_epochs):

        train_dataset = dataset.TrainDataset(train_files[epoch*args.num_file_train: (epoch+1)*args.num_file_train], static_map)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, collate_fn=dataset.collate_fn,  # noqa
                                    sampler=dataset.train_local_sampler(train_dataset), pin_memory=True)  # noqa

        length_dataloader = len(train_dataloader)
        model.train()
        running_loss = 0.0
        model.zero_grad()
        pbar = tqdm(enumerate(train_dataloader), total=length_dataloader, desc=f"Training {epoch+1}/{args.num_epochs}")
        for idx, data in pbar:

            inputs = data[0].to(args.device, dtype=torch.float)  # (bs, 105, 496, 448)
            targets = data[1].to(args.device, dtype=torch.float)  # (bs, 48, 495, 436)

            with amp.autocast():
                prediction = model(inputs)
                if args.use_mask:
                    mask_city = util.create_static_mask(args, static_map)
                    pred = prediction[:, :, 1:, 6:-6] * mask_city  # (bs, 105, 495, 436)
                else:
                    pred = prediction[:, :, 1:, 6:-6]  # (bs, 105, 495, 436)
                loss = criterian(pred, targets)

                if args.L1_regularization:
                    l1_weight = 0.001
                    l1 = l1_weight * sum(p.abs().sum() for p in model.parameters())
                    loss += l1

                loss = loss / args.accumulation_step

            scaler.scale(loss).backward()
            if ((idx + 1) % args.accumulation_step == 0) or (idx + 1 == length_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            global_step += 1
            dataset_size += inputs.shape[0]
            running_loss += (loss.item() * inputs.shape[0] * args.accumulation_step)
            epoch_loss = running_loss / dataset_size
            pbar.set_postfix(train_loss=f"{epoch_loss:0.5f}")

            # if scheduler is not None:
            #     scheduler.step(epoch_loss)

            if global_step % 8 == 0:
                writer.add_scalar("Training_loss/minibatches", epoch_loss, global_step)

            torch.cuda.empty_cache()
            gc.collect()

        train_epoch_loss = epoch_loss
        val_epoch_loss = util.evaluate(args, model, valid_dataloader, static_map, criterian, epoch, writer, experiment_name)

        current_lr = optimizer.param_groups[0]["lr"]

        if scheduler is not None:
            scheduler.step()

        util.log_to_tensorboard(writer, model, train_epoch_loss, val_epoch_loss, current_lr, epoch, save_histogram=True)

        print(f"Epoch: {epoch+1} | {train_epoch_loss=:0.5f} | {val_epoch_loss=:0.5f} | {current_lr=:0.7f}")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            if os.path.isdir(f"checkpoints/{experiment_name}"):
                torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                            f"checkpoints/{experiment_name}/best_model.pth")  # noqa
            else:
                os.makedirs(f"checkpoints/{experiment_name}")
                torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                            f"checkpoints/{experiment_name}/best_model.pth")  # noqa

    torch.save({'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict()},
               f"checkpoints/{experiment_name}/last_model.pth")


if __name__ == "__main__":

    start = time.time()

    args = parse_args()

    util.seed_everything(args.seed)

    util.make_dir()

    experiment_name = f"Training_{args.train_year[0]}_Validation_{args.val_year[0]}_{args.cities[0]}_{args.model}_unet512filter_L1_dropout"

    writer = SummaryWriter(f"logs/{experiment_name}")
    writer.add_text(experiment_name, f"scheduler: {args.scheduler}, wd: {args.wd}, L1: {args.L1_regularization}, lambda: 0.001, mask: {args.use_mask}, gr_acc: {args.accumulation_step}")

    print(f"Experiment(training): {experiment_name}\n")

    train(args, writer, experiment_name)

    print("Training_time (minute): ", round((time.time() - start)/60, 3))
