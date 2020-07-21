import torch.utils.data
from torch.nn import functional as F

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple




#用法：Conv2D_unitnorm(3, 64, 1, stride=1, bias = False)
class Conv2D_unitnorm(nn.Conv2d): #input:(batch, channel, Height, Weight)
    def forward(self, input):
        #self.weight = self.weight/torch.norm(self.weight)
        return F.conv2d(input, self.weight)
    def apply_constraint(self):  #拉到外面, 等到optimize之後再做
        self.weight = self.weight/torch.norm(self.weight)



#用法：Conv2d_samepadding(3,64,1,stride=1, padding='SAME')
class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 dilation, transposed, output_padding, groups, bias, weight=None):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(weight)

    def reset_parameters(self, weight):
        if weight == None:
            n = self.in_channels
            for k in self.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            self.weight.data.uniform_(-stdv, stdv)
        else:
            self.weight.data = torch.FloatTensor(weight)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

##for samepadding(without unit norm)
class Conv2d_samepadding(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=[1, 1],
                 padding='VALID', dilation=[1, 1], groups=1, bias=False, weight=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, weight)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride, \
                                   self.padding, self.dilation, self.groups)
    

    def apply_constraint(self):  #拉到外面, 等到optimize之後再做
        self.weight.data = self.weight.data/torch.norm(self.weight.data)


# custom con2d, because pytorch don't have "padding='same'" option.
def conv2d_same_padding(input, weight, bias=None, stride=1, padding='VALID', dilation=1, groups=1):
    def check_format(*argv):
        argv_format = []
        for i in range(len(argv)):
            if type(argv[i]) is int:
                argv_format.append((argv[i], argv[i]))
            elif hasattr(argv[i], "__getitem__"):
                argv_format.append(tuple(argv[i]))
            else:
                raise TypeError('all input should be int or list-type, now is {}'.format(argv[i]))

        return argv_format
    
    stride, dilation = check_format(stride, dilation)

    if padding == 'SAME':
        padding = 0

        input_rows = input.size(2)
        filter_rows = weight.size(2)
        out_rows = (input_rows + stride[0] - 1) // stride[0]
        padding_rows = max(0, (out_rows - 1) * stride[0] +
                            (filter_rows - 1) * dilation[0] + 1 - input_rows)
        rows_odd = padding_rows % 2

        input_cols = input.size(3)
        filter_cols = weight.size(3)
        out_cols = (input_cols + stride[1] - 1) // stride[1]
        padding_cols = max(0, (out_cols - 1) * stride[1] +
                            (filter_cols - 1) * dilation[1] + 1 - input_cols)
        cols_odd = padding_cols % 2

        input = pad(input, [padding_cols // 2, padding_cols // 2 + int(cols_odd),
                            padding_rows // 2, padding_rows // 2 + int(rows_odd)])
    
    elif padding == 'VALID':
        padding = 0
    
    elif type(padding) != int:        
        raise ValueError('Padding should be SAME, VALID or specific integer, but not {}.'.format(padding))
    return F.conv2d(input, weight, bias, stride, padding=padding,
                    dilation=dilation, groups=groups)



"""
class Conv2d_samepadding_unitnorm(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=[1, 1],
                 padding='VALID', dilation=[1, 1], groups=1, bias=False, weight=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, weight)

    def forward(self, input):
        return conv2d_same_padding_unitnorm(input, self.weight, self.bias, self.stride, \
                                   self.padding, self.dilation, self.groups)

    def apply_constraint(self):  #拉到外面, 等到optimize之後再做
        self.weight = self.weight/torch.norm(self.weight)




def conv2d_same_padding_unitnorm(input, weight, bias=None, stride=1, padding='VALID', dilation=1, groups=1):
    def check_format(*argv):
        argv_format = []
        for i in range(len(argv)):
            if type(argv[i]) is int:
                argv_format.append((argv[i], argv[i]))
            elif hasattr(argv[i], "__getitem__"):
                argv_format.append(tuple(argv[i]))
            else:
                raise TypeError('all input should be int or list-type, now is {}'.format(argv[i]))

        return argv_format
    
    stride, dilation = check_format(stride, dilation)

    if padding == 'SAME':
        padding = 0

        input_rows = input.size(2)
        filter_rows = weight.size(2)
        out_rows = (input_rows + stride[0] - 1) // stride[0]
        padding_rows = max(0, (out_rows - 1) * stride[0] +
                            (filter_rows - 1) * dilation[0] + 1 - input_rows)
        rows_odd = padding_rows % 2

        input_cols = input.size(3)
        filter_cols = weight.size(3)
        out_cols = (input_cols + stride[1] - 1) // stride[1]
        padding_cols = max(0, (out_cols - 1) * stride[1] +
                            (filter_cols - 1) * dilation[1] + 1 - input_cols)
        cols_odd = padding_cols % 2

        input = pad(input, [padding_cols // 2, padding_cols // 2 + int(cols_odd),
                            padding_rows // 2, padding_rows // 2 + int(rows_odd)])
    
    elif padding == 'VALID':
        padding = 0
    
    elif type(padding) != int:        
        raise ValueError('Padding should be SAME, VALID or specific integer, but not {}.'.format(padding))
    return F.conv2d(input, weight, bias, stride, padding=padding,
                    dilation=dilation, groups=groups)
"""







