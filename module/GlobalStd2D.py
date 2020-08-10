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
        self.min_std.data = self.min_std.data.clamp(0,float("inf"))
    #def compute_output_shape(self, input_shape):
        #return (input_shape[0], 1, 1, input_shape[-1]) 


class BackgroundStd2D(nn.Module):
    """
    Custom pytorch layer to compute sample-wise feature deviation
    Input:4維度(input：bf 而非devf5d)
    """
    def __init__(self, nb_feats, min_std_val=1e-5,**kwargs):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.min_std_val = torch.tensor(min_std_val, device=device)
        std_shape = (1, nb_feats, 1, 1)
        self.min_std = torch.nn.Parameter(torch.full(std_shape, self.min_std_val, device=device), requires_grad = True)

    def forward(self, bf, aspp_mask):
        
        aspp_mask_temp = torch.where(aspp_mask > 0.5, torch.zeros(aspp_mask.shape).cuda(), torch.ones(aspp_mask.shape).cuda()) #background = 1, foreground = 0
        stack = []
        for i in range(bf.shape[0]): #多少張圖片
            bf_std = bf[i,(aspp_mask_temp[i].expand(bf.shape[1],aspp_mask_temp.shape[2], aspp_mask_temp.shape[3]) !=0 )].view(1,bf.shape[1],-1) #(1, channel, H*W)
            bf_std = torch.std(bf_std, dim = 2)  #(1, channel)  
            bf_std = bf_std.view(1,-1,1,1)        
            stack.append(bf_std)
        x_std = torch.cat(stack)

        x_std = torch.max(x_std, self.min_std_val/10. + self.min_std)
        return x_std
    def apply_clamp(self):
        self.min_std.data = self.min_std.data.clamp(0,float("inf"))

