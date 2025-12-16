from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import numpy as np

from src.valid import valid

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def init_dataloader(dataset_name, 
                 transform,
                 batch_size = 64, 
                 dataset_load_path = 'data/',
                 train_mode = True, 
                 size = None):
    if dataset_load_path[-1] != '/':
        dataset_load_path += '/'
    dataset_name = dataset_name
    dataset_load_path = dataset_load_path + dataset_name + '_dataset'
    
    name2dataset = {
        'MNIST': datasets.MNIST,
        'FashionMNIST': datasets.FashionMNIST,
        'CIFAR10': datasets.CIFAR10,
        'CIFAR100': datasets.CIFAR100,
    }
    
    dataset = name2dataset[dataset_name](dataset_load_path,
                                download = True,
                                train = train_mode,
                                transform = transform)
    if size is not None and size != -1:
        dataset = torch.utils.data.Subset(dataset, np.arange(size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train_mode, num_workers = 2)
    return loader

def create_losses_func(dataloader, criterion):
    def calc_losses(model):
        losses, _ = valid(
            model,
            criterion,
            dataloader,
            model.device)
        return losses
    return calc_losses

def get_mean_and_std(dataloader):
    '''Compute the mean and std value of dataset.'''
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    length = 0
    for inputs, targets in dataloader:
        for i in range(3):
            for j in range(inputs.shape[0]):
                mean[i] += inputs[j,i,:,:].mean()
                std[i] += inputs[j,i,:,:].std()
                length += 1
    length/=3
    mean.div_(length)
    std.div_(length)
    return mean, std
