import torch
import torch.nn as nn

import numpy as np

class ConvBlock(nn.Module):
    r"""
        Convolutional block

        Args:
            c_in, c_out - input/output number of channels
            kernel_size - size of kernel
    """
    def __init__(self, 
            c_in : int, 
            c_out : int, 
            kernel_size : int, 
            padding : int,
            norm_class = None,
            dropout_p : float = 0.0):
        super().__init__()
        if norm_class is None:
            norm = nn.Identity()
        else:
            norm = norm_class(c_out)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, 
                kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            norm,
            nn.Dropout(dropout_p)
        )
    def forward(self, x):
        return self.body(x)

        
class ConvNet(nn.Module):
    r"""
    Convolutional network

    Args:
        channels_list list(int): list of layers number of channels
        input_sizes (int, int): width, height of input images
        classes (int): number of classes for classification

    """
    def __init__(self, 
                 layers_num : int, 
                 hidden_channels : int, 
                 kernel_size : int, 
                 padding : int,
                 input_channels : int, 
                 output_channels : int, 
                 input_sizes : tuple[int, int], 
                 classes : int,
                 norm_class = None,
                 dropout_p : float = 0.0):
        super().__init__()

        channels_list = [input_channels] + [hidden_channels for _ in range(layers_num)] + [output_channels]
        ker_size_list = [kernel_size for _ in channels_list[:-1]]
        self.model = self._stack_conv_blocks(
                channels_list, 
                ker_size_list, 
                padding,
                norm_class,
                dropout_p
        )
        self.flatten = nn.Flatten()

        flat_size = self._calc_flattened_size(channels_list, ker_size_list, input_sizes, padding)
        self.head = nn.Sequential(
            nn.Linear(flat_size, classes)
        )

    def _calc_flattened_size(self, channels_list, ker_size_list, input_sizes, padding):
        ker_delta = np.sum(np.array(ker_size_list)-1)
        pad_delta = padding*2*(len(ker_size_list)-1)
        final_sizes = (input_sizes[0] - ker_delta + pad_delta, input_sizes[1] - ker_delta + pad_delta)
        return final_sizes[0]*final_sizes[1]*channels_list[-1]

    def _stack_conv_blocks(self, 
            channels_list : list[int],
            ker_size_list : list[int],
            padding : int,
            norm_class = None,
            dropout_p : float = 0.0):

        convs_list = []
        for c_in, c_out, kernel_size in zip(channels_list[:-2], channels_list[1:-1], ker_size_list[:-1]):
            convs_list.append(ConvBlock(c_in, c_out, kernel_size, padding, norm_class, dropout_p))
        convs_list.append(nn.Conv2d(channels_list[-2], channels_list[-1], ker_size_list[-1], padding))
        return nn.Sequential(*convs_list)

    def forward(self, images):
        out = self.model(images)
        flattened = self.flatten(out)
        logits = self.head(flattened)
        return logits

    @property
    def device(self):
        return next(iter(self.parameters())).device
