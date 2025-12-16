import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets
import torch
import torch.nn as nn

import copy
import json
import argparse
import hydra
from omegaconf import OmegaConf, DictConfig
import pickle
import torch.distributed as dist
from torch.multiprocessing import Process
import random
import os

from src.models.mlp import MLP
from src.models.conv import ConvNet
from src.utils import init_dataloader
from src.utils import smooth
from src.train import train
from src.valid import valid
from src.visualize import calc_grid_loss
from src.calc_delta import DeltaCalculator


def init_process(rank, size, fn, master_port, config, backend="nccl"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(config)


def main_logger(
    config,
    model,
    model_kwargs,
    full_train_losses,
    train_loader,
    val_loader,
    criterion,
    num_trained_epochs,
    device,
    setup_id: str,
):
    val_losses, val_acc = valid(model, criterion, val_loader, device=device)
    train_losses, train_acc = valid(model, criterion, train_loader, device=device)

    delta_calc = DeltaCalculator(model, train_loader, criterion)
    deltas = delta_calc.calc_deltas(**config["delta_vis_params"])
    deviations = deltas * np.arange(2, len(deltas) + 2)

    log_path = config["logging"]["save_log_path"]
    try:
        os.makedirs(f"{log_path}/{setup_id}")  # create folder for model results
    except OSError:
        print('cannot create')
    with open(f"{log_path}/{setup_id}/deviations_.pickle", "wb") as f:
        pickle.dump(deviations, f)
    with open(f"{log_path}/{setup_id}/train_losses_.pickle", "wb") as f:
        pickle.dump(train_losses, f)
    with open(f"{log_path}/{setup_id}/val_losses_.pickle", "wb") as f:
        pickle.dump(val_losses, f)
    with open(f"{log_path}/{setup_id}/train_acc_.pickle", "wb") as f:
        pickle.dump(train_acc, f)
    with open(f"{log_path}/{setup_id}/val_acc_.pickle", "wb") as f:
        pickle.dump(val_acc, f)
    with open(f"{log_path}/{setup_id}/full_train_losses_.pickle", "wb") as f:
        pickle.dump(full_train_losses, f)

    log_data = dict()

    log_data['class'] = config['model']['class']
    log_data["train_loss"] = np.mean(train_losses)
    log_data["val_loss"] = np.mean(val_losses)
    log_data["train_acc"] = np.mean(train_acc)
    log_data["val_acc"] = np.mean(val_acc)
    log_data["deviation"] = np.mean(deviations)
    log_data["save_log_path"] = f"{log_path}/{setup_id}"
    log_data['delta_vis_params'] = OmegaConf.to_container(config['delta_vis_params'], resolve = True)

    # data logs
    log_data["num_epochs"] = num_trained_epochs
    log_data["loader_size"] = config['dataset']['size']

    # model params log
    for save_param in config["logging"]["save_params"]:
        log_data[save_param] = model_kwargs[save_param]
    log_data['num_params'] = sum([p.numel() for p in model.parameters()])

    with open(f"{log_data['save_log_path']}/main.json", "w") as outfile:
        json.dump(log_data, outfile, indent=4)

def create_setup(config):
    torch.manual_seed(dist.get_rank())
    cuda_devices = config["distributed"]["cuda_devices"]
    device = torch.device(
        torch.device("cuda", cuda_devices[dist.get_rank() % len(cuda_devices)])
    )
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                config["dataset"]["transform_params"]["mean"],
                config["dataset"]["transform_params"]["std"],
            ),
        ]
    )
    train_loader = init_dataloader(
        dataset_name=config["dataset"]["name"],
        transform=transform,
        batch_size=config["dataset"]["batch_size"],
        dataset_load_path=config["dataset"]["load_path"],
        size=config["dataset"]["size"],
    )
    val_loader = init_dataloader(
        dataset_name=config["dataset"]["name"],
        transform=transform,
        batch_size=config["dataset"]["batch_size"],
        dataset_load_path=config["dataset"]["load_path"],
        size=-1,
        train_mode=False,
    )
    cnt = 0
    for variable_param in config["model"][config["model"]["variable_param"]]:
        for num_epochs in config["train_params"]["num_epochs"]:

            kwargs = dict(copy.deepcopy(config["model"]))
            del kwargs["class"]
            del kwargs["variable_param"]
            kwargs[config["model"]["variable_param"]] = variable_param

            model = ConvNet(**kwargs).to(device)
            criterion = nn.CrossEntropyLoss()
            if config["train_params"]["optimizer"] == "Adam":
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=config["train_params"]["learning_rate"]
                )
            else:
                # TODO
                print("provide optimizer!!!")
                return

            model, train_losses = train(
                model,
                optimizer,
                criterion,
                train_loader,
                num_epochs,
                device=device,
            )
            setup_id = f"{dist.get_rank()}_{cnt}"
            main_logger(
                config,
                model,
                kwargs,
                train_losses,
                train_loader,
                val_loader,
                criterion,
                num_epochs,
                device,
                setup_id,
            )
            cnt += 1


@hydra.main(version_base=None, config_path="configs/full_try_exps", config_name="3_1")
def main(config: DictConfig):
    size = config["num_exps"]
    processes = []
    port = random.randint(25000, 30000)
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, create_setup, port, config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
