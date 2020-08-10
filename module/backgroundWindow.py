import torch
import torch.nn as nn
import torch.nn.functional as F


class backgroundWindow(nn.Module):
    def __init__(self, window_size = [7,15,31]):
        super().__init__()
        self.window_size = window_size

    def forward(self, bf, aspp_mask):

        mean_mask = []
        for w in [7, 15, 31]:
            avg_pool = nn.AvgPool2d(w, stride=1)
            padding = int((w - 1)/2)

            #取threshold
            aspp_mask_temp = torch.where(aspp_mask > 0.5, torch.zeros(aspp_mask.shape).cuda(), torch.ones(aspp_mask.shape).cuda()) #background = 1, foreground = 0
            bf_binary = bf * aspp_mask_temp #只留background的值  #####要對它做mean(這時已經把ground truth忽略掉了)

            #取threshold後的mean,
            bf_binary = F.pad(bf_binary, (padding, padding, padding, padding),"constant",value = 0)
            aspp_mask_temp = F.pad(aspp_mask, (padding, padding, padding, padding),"constant",value = 1)
            reciprocal = torch.where(aspp_mask_temp >0, torch.zeros(bf_binary.shape).cuda(), torch.ones(bf_binary.shape).cuda() ) #大於0 foreground+pad
            reciprocal = avg_pool(reciprocal)
            reciprocal = torch.reciprocal(reciprocal, out=None)
            reciprocal[torch.isinf(reciprocal)] = 0  
            bf_mean = reciprocal * avg_pool(bf_binary)
            mean_mask.append(torch.abs(bf - bf_mean))
        #global
        bf_mean = torch.mean(bf, dim = (2,3), keepdim = True) * torch.ones_like(bf)
        mean_mask.append(torch.abs(bf - bf_mean))
        devf5d = torch.stack(mean_mask, dim=1)  
        return devf5d