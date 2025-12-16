import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

def valid(model,
          criterion,
          dataloader,
          device):

        losses = []
        acc = []

        model.eval()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            losses.append(loss.detach().cpu().item())
            
            acc.append(torch.mean((torch.argmax(outputs, dim = -1) == y).to(torch.float)).cpu().item())

        return losses, acc
