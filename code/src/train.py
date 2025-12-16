import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
from src.valid import valid
from collections import defaultdict


def train(model,
          optimizer,
          criterion,
          train_dataloader,
          num_epochs,
          device):
    model.train()
    # Train the network
    losses = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().item())
    return model, losses


def train_and_log(
        model,
        optimizer,
        criterion,
        train_dataloader,
        valid_dataloader,
        num_epochs,
        device):

    train_metrics = defaultdict(list)
    valid_metrics = defaultdict(list)

    for epoch in range(num_epochs):
        model, _ = train(model, optimizer, criterion,
                              train_dataloader, 1, device)
        train_losses, train_accs = valid(model, criterion, train_dataloader, device)        
        valid_losses, valid_accs = valid(model, criterion, valid_dataloader, device)

        train_metrics['loss'].append(np.mean(train_losses))
        valid_metrics['loss'].append(np.mean(valid_losses))

        train_metrics['accuracy'].append(np.mean(train_accs))
        valid_metrics['accuracy'].append(np.mean(valid_accs))
    
    return model, train_metrics, valid_metrics