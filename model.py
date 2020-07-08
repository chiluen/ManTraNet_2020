import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from module import Conv2d_modified, ConvLSTM, create_featex, NestedWindow, GlobalStd2D


##替代 create_manTraNet_model功能
class ManTraNet(nn.Module):
    def __init__(self, Featex, pool_size_list=[7,15,31], is_dynamic_shape=True, apply_normalization=True):
        super().__init__()
        self.Featex = Featex
        self.pool_size_list = pool_size_list
        self.is_dynamic_shape = is_dynamic_shape
        self.apply_normalization = apply_normalization

        #layers
        self.outlierTrans = Conv2d_modified.Conv2d_samepadding_unitnorm(256, 64, 1, padding='SAME') 
        self.pred = Conv2d_modified.Conv2d_samepadding(8, 1, 7, padding="SAME")  ##還沒做更改
        self.bnorm = nn.BatchNorm2d(64)
        self.nestedAvgFeatex = NestedWindow.NestedWindowAverageFeatExtrator(window_size_list= self.pool_size_list, 
                                                                               output_mode='5d',
                                                                               minus_original=True) 
        self.glbStd = GlobalStd2D.GlobalStd2D(64)   #input: number of features
        self.cLSTM = ConvLSTM.ConvLSTM(input_dim = 64, hidden_dim = 8, kernel_size = (7, 7), num_layers = 1, batch_first = True, bias = True, return_all_layers = False)

    def forward(self,x):
        rf = self.Featex(x) 
        rf = self.outlierTrans(rf) 
        bf = self.bnorm(rf) #(batch, channel=64, H, W)
        devf5d = self.nestedAvgFeatex(bf) #(batch, 4, channel=64, H, W)
        if self.apply_normalization:
            sigma = self.glbStd(bf) #(batch, channel, H, W)
            sigma5d = torch.unsqueeze(sigma, 1) 
            devf5d = torch.abs(devf5d / sigma5d) 

        # Convert back to 4d
        _, last_states = self.cLSTM(devf5d)  #(batch, channel = 8, H, W)
        devf = last_states[0][0] #(batch, channel = 8, H, W)
        pred_out = self.pred(devf) #(batch, channel = 1, H, W)
        return pred_out


def create_model(IMC_model_idx, freeze_featex, window_size_list=[7,15,31]):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    type_idx = IMC_model_idx if IMC_model_idx < 4 else 2
    Featex = create_featex.Featex_vgg16_base(type_idx).to(device)
    if freeze_featex:
        print("INFO: freeze feature extraction part, trainable=False")
        #Featex.trainable = False ##它沒有trainable這個選項阿？？？  
    else:
        print ("INFO: unfreeze feature extraction part, trainable=True")
    """
    if len(window_size_list) == 4:
        for ly in Featex.layers[:5]:
            ly.trainable = False  ##也沒有這個選項
            print("INFO: freeze", ly.name)
    """
    model = ManTraNet(Featex, pool_size_list=window_size_list, is_dynamic_shape=True, apply_normalization=True)
    model.to(device)
    return model




