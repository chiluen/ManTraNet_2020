import numpy as np
import torch.nn as nn

def _pad_symmetric(input, padding):
    # padding is left, right, top, bottom
    in_sizes = input.size()

    x_indices = [i for i in range(in_sizes[-1])]  # [0, 1, 2, 3, ...]
    left_indices = [i for i in range(padding[0] - 1, -1, -1)]  # e.g. [3, 2, 1, 0]
    right_indices = [-(i + 1) for i in range(padding[1])]  # e.g. [-1, -2, -3]
    x_indices = torch.tensor(left_indices + x_indices + right_indices)

    y_indices = [i for i in range(in_sizes[-2])]
    top_indices = [i for i in range(padding[2] - 1, -1, -1)]
    bottom_indices = [-(i + 1) for i in range(padding[3])]
    y_indices = torch.tensor(top_indices + y_indices + bottom_indices)

    ndim = input.ndim
    if ndim == 3:
        return input[:, y_indices[:, None], x_indices[None, :]]
    elif ndim == 4:
        return input[:, :, y_indices[:, None], x_indices[None, :]]
    else:
        raise RuntimeError("Symmetric padding of N-D tensors are not supported yet")


class Conv2DSymPadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=(1, 1), dilation=(1, 1), bias=True):

        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
        else:
            kh = kw = kernel_size
        ph, pw = kh//2, kw//2

        super(Conv2DSymPadding, self).__init__(in_channels, out_channels, kernel_size,
                                               stride=stride, dilation=dilation, bias=bias, 
                                               padding=(ph, pw))

    def _conv_forward(self, input, weight):
        return F.conv2d(_pad_symmetric(input, self._reversed_padding_repeated_twice),
                        weight, self.bias, self.stride,
                        _pair(0), self.dilation, self.groups)