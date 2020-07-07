import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import Conv2d_modified
import ConvLSTM
import create_featex





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


#這不需要
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__():
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

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

class BayarConstraint() :
    def __init__(self):
        self.mask = None
    def _initialize_mask( self, w):
        out_channels, in_channels, kernel_height, kernel_width = w.size()
        m = np.zeros((out_channels, in_channels, kernel_height, kernel_width)).astype('float32')
        m[:, :, kernel_height//2, kernel_width//2] = 1.
        self.mask = torch.tensor(m)
        return
    def __call__(self, w) :
        if self.mask is None :
            self._initialize_mask(w)
        w *= (1-self.mask)
        rest_sum = torch.sum( w, dim=(-1, -2), keepdim=True)
        w /= rest_sum + 1e-7 #K.epsilon() = 1e-7
        w -= self.mask
        return w

class CombinedConv2D(Conv2DSymPadding):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5),
                 stride=(1, 1), dilation=(1, 1), bias=False):
        super(CombinedConv2D, self).__init__(in_channels, out_channels, kernel_size,
                                             stride=stride, dilation=dilation, bias=bias)
        del self.weight
        self._build_all_kernel()
        self.bayar_constraint = BayarConstraint()
    
    def _get_srm_list(self):
        # srm kernel 1                                                                                                                                
        srm1 = np.zeros([5,5]).astype('float32')
        srm1[1:-1,1:-1] = np.array([[-1, 2, -1],
                                    [2, -4, 2],
                                    [-1, 2, -1]] )
        srm1 /= 4.
        # srm kernel 2                                                                                                                                
        srm2 = np.array([[-1, 2, -2, 2, -1],
                         [2, -6, 8, -6, 2],
                         [-2, 8, -12, 8, -2],
                         [2, -6, 8, -6, 2],
                         [-1, 2, -2, 2, -1]]).astype('float32')
        srm2 /= 12.
        # srm kernel 3                                                                                                                                
        srm3 = np.zeros([5,5]).astype('float32')
        srm3[2,1:-1] = np.array([1,-2,1])
        srm3 /= 2.
        return [ srm1, srm2, srm3 ]
    
    # torch -> C_out, C_in, kH, kW
    def _build_SRM_kernel(self):
        kernel = []
        srm_list = self._get_srm_list()
        for idx, srm in enumerate(srm_list):
            for ch in range(3):
                this_ch_kernel = np.zeros([3,5,5]).astype('float32')
                this_ch_kernel[ch,:,:] = srm
                kernel.append(this_ch_kernel)
        kernel = np.stack(kernel, axis=0)
        srm_kernel = nn.Parameter(torch.tensor(kernel), requires_grad=False)
        return srm_kernel

    def _build_all_kernel(self):
        # 1. regular conv kernels, fully trainable
        out_channels = self.out_channels - 9 - 3
        if out_channels >= 1:
            regular_kernel_shape = (out_channels, self.in_channels) + self.kernel_size
            self.regular_kernel = nn.Parameter(torch.ones(regular_kernel_shape))
            nn.init.xavier_uniform_(self.regular_kernel.data)
        else:
            self.regular_kernel = None
        # 2. SRM kernels, not trainable
        self.srm_kernel = self._build_SRM_kernel()
        # 3. bayar kernels, trainable but under constraint
        bayar_kernel_shape = (3, self.in_channels) + self.kernel_size
        self.bayar_kernel = nn.Parameter(torch.ones(bayar_kernel_shape))
        nn.init.xavier_uniform_(self.bayar_kernel.data)
    
    def apply_bayar_constraint(self):
        self.bayar_constraint(self.bayar_kernel.data)
        return
    
    def forward(self, input):
        if self.regular_kernel is not None:
            regular_out = super(CombinedConv2D, self)._conv_forward(input, self.regular_kernel)
        srm_out = super(CombinedConv2D, self)._conv_forward(input, self.srm_kernel)
        bayar_out = super(CombinedConv2D, self)._conv_forward(input, self.bayar_kernel)
        
        if self.regular_kernel is not None:
            all_outs = [regular_out, 
                        srm_out,
                        bayar_out]
        else:
            all_outs = [srm_out,
                        bayar_out]
        
        return torch.cat(all_outs, dim=1)

class GlobatStd2D(nn.Module):
    """
    Custom pytorch layer to compute sample-wise feature deviation
    """
    def __init__(self, input_shape, min_std_val=1e-5,**kwargs):
        super().__init__()
        self.min_std_val = min_std_val
        
        #build min_std部份
        nb_feats = input_shape[-1]
        std_shape = (1,1,1,nb_feats)
        self.min_std_unconstraint = torch.nn.Parameter(torch.full(std_shape, self.min_std_val), requires_grad = True) #先沒加上constraint
        self.min_std = self.min_std_unconstraint.data.clamp(0,float("inf"))
    def forward(self, x):
        x_std = torch.std(x, dim = (1,2), keepdim=True)
        x_std = torch.max(x_std, self.min_std_val/10. + self.min_std)
        return x_std
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, 1, input_shape[-1])

class NestedWindowAverageFeatExtrator(nn.Module) :
    '''Custom Keras Layer of NestedWindowAverageFeatExtrator
    '''
    def __init__( self,
                  window_size_list,
                  output_mode='5d',
                  minus_original=False,
                  include_global=True,
                  **kwargs ) :
        '''
        INPUTS:
            win_size_list = list of int or tuples, each elem indicate a winsize of interest
            output_mode = '5d' or '4d', where
                          '5d' merges all win_avgs along a new time axis
                          '4d' merges all win_avgs along the existing feat axis
        '''
        self.window_size_list = window_size_list
        assert output_mode in ['5d','4d'], "ERROR: unkown output mode={}".format( output_mode )
        self.output_mode = output_mode
        self.minus_original = minus_original
        self.include_global = include_global
        super(NestedWindowAverageFeatExtrator, self).__init__(**kwargs)
    def build( self, input_shape ) :
        self.num_woi = len( self.window_size_list )
        self.count_ii = None
        self.lut = dict()
        self.built = True
        self.max_wh, self.max_ww = self._get_max_size()
        return
    def _initialize_ii_buffer( self, x ) :
        x_pad = F.pad(x, (self.max_wh//2+1, self.max_wh//2+1, self.max_ww//2+1, self.max_ww//2+1),"constant",value = 0)
        ii_x  = torch.cumsum( x_pad, axis = 2) # height
        ii_x2 = torch.cumsum( ii_x, axis = 3) # width
        return ii_x2
    def _get_max_size( self ) :
        mh, mw = 0, 0
        for hw in self.window_size_list :
            if ( isinstance( hw, int ) ) :
                h = w = hw
            else :
                h, w = hw[:2]
            mh = max( h, mh )
            mw = max( w, mw )
        return mh, mw
    def _compute_for_one_size( self, x, x_ii, height, width ) :
        # 1. compute valid counts for this key
        top   = self.max_wh//2 - height//2
        bot   = top + height
        left  = self.max_ww//2 - width //2
        right = left + width
        Ay, Ax = (top, left) #self.max_wh, self.max_ww
        By, Bx = (top, right) # Ay, Ax + width
        Cy, Cx = (bot, right) #By + height, Bx
        Dy, Dx = (bot, left) #Cy, Ax
        ii_key = (height,width)
        top_0   = -self.max_wh//2 - height//2 - 1
        bot_0   = top_0 + height
        left_0  = -self.max_ww//2 - width//2 - 1
        right_0 = left_0 + width
        Ay0, Ax0 = (top_0, left_0) #self.max_wh, self.max_ww
        By0, Bx0 = (top_0, right_0) # Ay, Ax + width
        Cy0, Cx0 = (bot_0, right_0) #By + height, Bx
        Dy0, Dx0 = (bot_0, left_0) #Cy, Ax
        # used in testing, where each batch is a sample of different shapes
        counts = torch.ones_like( x[:1,...,:1] )
        count_ii = self._initialize_ii_buffer( counts )
        # compute winsize if necessary
        counts_2d = count_ii[:,Ay:Ay0, Ax:Ax0] \
                  + count_ii[:,Cy:Cy0, Cx:Cx0] \
                  - count_ii[:,By:By0, Bx:Bx0] \
                  - count_ii[:,Dy:Dy0, Dx:Dx0]
        # 2. compute summed feature
        sum_x_2d = x_ii[:,Ay:Ay0, Ax:Ax0] \
                 + x_ii[:,Cy:Cy0, Cx:Cx0] \
                 - x_ii[:,By:By0, Bx:Bx0] \
                 - x_ii[:,Dy:Dy0, Dx:Dx0]
        # 3. compute average feature
        avg_x_2d = sum_x_2d / counts_2d
        return avg_x_2d
    def forward( self, x ) :
        x_win_avgs = []
        # 1. compute corr(x, window_mean) for different sizes
        # 1.1 compute integral image buffer
        x_ii = self._initialize_ii_buffer( x )
        for hw in self.window_size_list :
            if isinstance( hw, int ) :
                height = width = hw
            else :
                height, width = hw[:2]
            this_avg = self._compute_for_one_size( x, x_ii, height, width )
            if ( self.minus_original ) :
                x_win_avgs.append( this_avg-x )
            else :
                x_win_avgs.append( this_avg )
        # 2. compute corr(x, global_mean)
        if ( self.include_global ) :
            if ( self.minus_original ) :
                mu = torch.mean(x, dim = (2,3), keepdim = True)
                x_win_avgs.append( mu-x )
            else :
                mu = torch.mean(x, dim = (2,3), keepdim = True) * torch.ones_like(x)
                x_win_avgs.append( mu )
        if self.output_mode == '4d' :
            # concate channel
            return torch.cat( x_win_avgs, axis= 1)
        elif self.output_mode == '5d' :
            # stack num_woi
            return torch.stack( x_win_avgs, axis= 1)
        else :
            raise (NotImplementedError, "ERROR: unknown output_mode={}".format( self.output_mode ))
    def compute_output_shape(self, input_shape):
        batch_size, num_rows, num_cols, num_filts = input_shape
        if self.output_mode == '4d' :
            return ( batch_size, num_rows, num_cols, (self.num_woi+int(self.include_global))*num_filts )
        else :
            return ( batch_size, self.num_woi+int(self.include_global), num_rows, num_cols, num_filts )



##替代 create_manTraNet_model功能
class ManTraNet(nn.Module):
    def __init__(self, Featex, pool_size_list=[7,15,31], is_dynamic_shape=True, apply_normalization=True):
        self.Featex = Featex
        self.pool_size_list = pool_size_list
        self.is_dynamic_shape = is_dynamic_shape
        self.apply_normalization = apply_normalization

        #layers
        self.Conv2D_padding_constraint = Conv2d_modified.Conv2d_samepadding_unitnorm(256, 64, 1, padding='SAME') #input dimension要再確認, 我沒有填寫 
        self.Conv2D_padding = Conv2d_modified.Conv2d_samepadding(999, 1, 7, padding="SAME")
        self.Batchnorm = nn.BatchNorm2d(64)
        self.NestedWindowAverageFeatExtrator = NestedWindowAverageFeatExtrator() ###這個要去對照
        self.GlobalStd2D = GlobatStd2D(???)   ###input_shape是什麼？
        self.Lambda = LambdaLayer()
        self.ConvLSTM = ConvLSTM() ##這個input?

    def forward(self,x):
        rf = self.Featex(x)  ##這邊Featex還沒確定可不可以接起來
        rf = self.Conv2D_padding_constraint(rf)
        bf = self.Batchnorm(rf)
        devf5d = self.NestedWindowAverageFeatExtrator(bf)
        if apply_normalization:
            sigma = GlobalStd2D(bf)
            sigma5d = torch.unsqueeze(sigma, 1) #我把lambda改掉
            devf5d = torch.abs(devf5d / sigma5d) #我把lambda改掉

        # Convert back to 4d
        devf = self.ConvLSTM(devf5d)
        pred_out = self.Conv2D_padding(devf)
        return pred_out


def create_model(IMC_model_idx, freeze_featex, window_size_list):
    type_idx = IMC_model_idx if IMC_model_idx < 4 else 2
    Featex = create_featex.create_featex_vgg16_base(type_idx)
    if freeze_featex:
        print("INFO: freeze feature extraction part, trainable=False")
        Featex.trainable = False ##它沒有trainable這個選項阿？？？
    else:
        print ("INFO: unfreeze feature extraction part, trainable=True")
    if len(window_size_list) == 4:
        for ly in Featex.layers[:5]:
            ly.trainable = False
            print("INFO: freeze", ly.name)
    model = ManTraNet(Featex, pool_size_list=window_size_list, is_dynamic_shape=True, apply_normalization=True)
    return model




