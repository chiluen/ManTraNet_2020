import torch
import torch.nn as nn


class GlobalStd2D(nn.Module):
    """
    Custom pytorch layer to compute sample-wise feature deviation
    Input:4維度(input：bf 而非devf5d)
    """
    def __init__(self, nb_feats, min_std_val=1e-5,**kwargs):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.min_std_val = torch.tensor(min_std_val, device=device)
        
        #build min_std部份
        #nb_feats = input_shape[-3] # for pytorch "channel"
        std_shape = (1, nb_feats, 1, 1)
        self.min_std = torch.nn.Parameter(torch.full(std_shape, self.min_std_val, device=device), requires_grad = True)
        #self.min_std_clamp = self.min_std.data.clamp(0,float("inf"))
    def forward(self, x):
        x_std = torch.std(x, dim = (2,3), keepdim=True) 
        x_std = torch.max(x_std, self.min_std_val/10. + self.min_std)
        return x_std
    def apply_clamp(self):
        self.min_std = self.min_std.data.clamp(0,float("inf"))
    #def compute_output_shape(self, input_shape):
        #return (input_shape[0], 1, 1, input_shape[-1]) 