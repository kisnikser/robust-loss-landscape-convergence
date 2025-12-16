from src.visualize import LossVisualizer

from src.models.mlp import MLP
from src.models.conv import ConvNet

from src.utils import init_dataloader
from src.utils import smooth
from src.train import train
from src.valid import valid

from src.visualize import calc_grid_loss 
from src.calc_delta import DeltaCalculator

import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets
import torch
import torch.nn as nn

import copy
import json
import argparse
from omegaconf import OmegaConf
import hydra
from omegaconf import OmegaConf, DictConfig

myparams = {
    'font.size': 16,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
}
plt.rcParams.update(myparams)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

@hydra.main(version_base=None, config_path='configs/CIFAR10', config_name='conv_channels_sigm8')
def main(config: DictConfig):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config['dataset']['transform_params']['mean'], 
                             config['dataset']['transform_params']['std'])
    ])

    dataloader = init_dataloader(
        dataset_name=config['dataset']['name'],
        transform=transform,
        batch_size=config['dataset']['batch_size'],
        dataset_load_path=config['dataset']['load_path'],
        size=config['dataset']['size']
    )

    if config['model']['class'] == 'conv':
        model_class = ConvNet
    elif config['model']['class'] == 'mlp':
        model_class = MLP
    
    param_to_deltas = {}
    param_to_final_loss = {}
    param_to_train_losses = {}
    param_to_cummean_loss = {}

    for param in config['model']['variable_param_grid']:
        kwargs = dict(copy.deepcopy(config['model']))
        del kwargs['class']; del kwargs['variable_param']; del kwargs['variable_param_grid']
        kwargs[config['model']['variable_param']] = param

        model = model_class(**kwargs).to(config['device'])
        if config['train_params']['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = config['train_params']['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        model, train_losses = train(
            model,
            optimizer,
            criterion,
            dataloader,
            num_epochs=config['train_params']['num_epochs'],
            device = config['device']
        )

        delta_calc = DeltaCalculator(model, dataloader, criterion)
        deltas = delta_calc.calc_deltas(**config['delta_vis_params'])

        param_to_deltas[param] = deltas
        final_losses = np.array(valid(model, criterion, dataloader, device=config['device'])[0])
        param_to_final_loss[param] = np.mean(final_losses)
        param_to_train_losses[param] = train_losses
        param_to_cummean_loss[param] = np.cumsum(final_losses)/np.arange(1, len(final_losses)+1)

    mean_deltas = {i:np.mean(param_to_deltas[i][:-1]*np.arange(1, len(param_to_deltas[i]))) for i in param_to_deltas.keys()}
    fig, axs = plt.subplots(ncols=2, nrows=3, figsize = (12, 12))
    suptitle = ""
    for key, value in config['delta_vis_params'].items():
        suptitle += f'{key}:{value}\n'
    fig.suptitle(suptitle)
    for i in param_to_deltas.keys():
        arr = param_to_deltas[i]*np.arange(1, len(param_to_deltas[i])+1)
        axs[0][0].plot(smooth(arr, config['visualize']['smooth_window']), label = f'{i}')
        axs[0][0].plot(arr)
        axs[1][0].plot(smooth(arr/param_to_cummean_loss[i][1:], config['visualize']['smooth_window']), label = f'{i}')
        axs[1][0].plot(arr/param_to_cummean_loss[i][1:])
        arr = param_to_train_losses[i]
        arr = smooth(arr, config['visualize']['train_loss_smooth_window'])
        axs[2][0].plot(arr, label = f'{i}')
    
    axs[0][0].set(
        ylabel = '$|k*\Delta_k|$',
        xlabel = 'k'
    )
    axs[2][0].set(
        ylabel = "$\mathcal{L}$",
        xlabel = 'k'
    )
    axs[2][0].set_xlim(left = config['visualize']['train_loss_smooth_window'])

    axs[1][0].set(
        ylabel = '$|k*\Delta_k|$/$\mathcal{L}_k$',
        xlabel = 'k'
    )

    y, x = np.array(list(mean_deltas.values())), list(mean_deltas.keys())
    axs[0][1].plot(x, y, label = 'avg mean $k\Delta_k$')

    axs[0][1].set(
        xticks = x,
        ylabel = 'avg($k\Delta_k$)',
        xlabel = config['model']['variable_param']
    )

    y = [np.mean(param_to_deltas[i]*np.arange(1, len(param_to_deltas[i])+1)/param_to_cummean_loss[i][1:]) for i in param_to_deltas.keys()]
    axs[1][1].plot(x, y)
    axs[1][1].set(
        xticks = x,
        ylabel = 'avg($k\Delta_k$/$\mathcal{L}_k)$',
        xlabel = config['model']['variable_param']
    )

    axs[2][1].plot(x, np.array(list(param_to_final_loss.values())))
    axs[2][1].set(
        xticks = x,
        ylabel = 'final train loss',
        xlabel = config['model']['variable_param']
    )


    axs[0][0].legend(title = config['model']['variable_param'])
    axs[1][0].legend(title = config['model']['variable_param'])
    axs[2][0].legend(title = config['model']['variable_param'])

    plt.savefig(config['logging']['savefig_path'])

main()
