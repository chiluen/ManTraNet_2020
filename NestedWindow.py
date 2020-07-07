import torch.nn as nn
import torch
import torch.nn.functional as F

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
        self.num_woi = len( self.window_size_list )
        self.count_ii = None
        self.lut = dict()
        self.built = True
        self.max_wh, self.max_ww = self._get_max_size()
        super(NestedWindowAverageFeatExtrator, self).__init__(**kwargs)

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
        counts = torch.ones_like( x[:1, :1, ...] )
        count_ii = self._initialize_ii_buffer( counts )
        # compute winsize if necessary
        counts_2d = count_ii[:,:,Ay:Ay0, Ax:Ax0] \
                  + count_ii[:,:,Cy:Cy0, Cx:Cx0] \
                  - count_ii[:,:,By:By0, Bx:Bx0] \
                  - count_ii[:,:,Dy:Dy0, Dx:Dx0]
        # 2. compute summed feature
        sum_x_2d = x_ii[:,:,Ay:Ay0, Ax:Ax0] \
                 + x_ii[:,:,Cy:Cy0, Cx:Cx0] \
                 - x_ii[:,:,By:By0, Bx:Bx0] \
                 - x_ii[:,:,Dy:Dy0, Dx:Dx0]
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