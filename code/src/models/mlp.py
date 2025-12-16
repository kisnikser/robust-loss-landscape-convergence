import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    r"""
    Multi-Layer Perceptron (MLP) model.

    Args:
        sizes_list list(int): list of layers sizes
        activation_class: activation after all linear layers
    """
    def __init__(self,
            layers_num : int,
            hidden : int,
            input_channels : int,
            input_sizes : int, 
            classes : int, 
            norm_class = None,
            dropout_p : float = 0.0):
        super().__init__()
        sizes_list = [np.prod(input_sizes)*input_channels] + [hidden]*layers_num + [classes]
        self.layers = []

        activation_class = nn.ReLU
        for in_size, out_size in zip(sizes_list[:-2], sizes_list[1:-1]):
            if norm_class is None:
                norm_layer = nn.Identity()
            else:
                norm_layer = norm_class(out_size)
            self.layers.append(nn.Sequential(
                nn.Linear(in_size, out_size),
                activation_class(),
                norm_layer,
                nn.Dropout(dropout_p)
            ))
        self.layers.append(nn.Linear(sizes_list[-2], sizes_list[-1]))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.layers(x)
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
