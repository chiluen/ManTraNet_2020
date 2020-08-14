import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from module import Conv2d_modified, ConvLSTM, create_featex, NestedWindow, GlobalStd2D, ASPP, AFNB_one_input
import load_weights
import ipdb

##替代 create_manTraNet_model功能
class ManTraNet(nn.Module):
    def __init__(self, Featex, pool_size_list=[7,15,31], is_dynamic_shape=True, apply_normalization=True, OPTS_aspp = False):
        super().__init__()
        self.Featex = Featex
        self.pool_size_list = pool_size_list
        self.is_dynamic_shape = is_dynamic_shape
        self.apply_normalization = apply_normalization
        first_channel = 128
        second_channel = 64
        in_channels = [128, 256]
        key_value_channels = [128,128]

        self.ann_nl_1 = AFNB_one_input.SelfAttentionBlock2D(in_channels = 256, out_channels=256, 
                                            key_channels=128, value_channels=128)
        
        self.ann_nl_2 = AFNB_one_input.SelfAttentionBlock2D(in_channels = 64, out_channels=64, 
                                            key_channels=32, value_channels=32)


        self.conv_1 = nn.Sequential( 
            Conv2d_modified.Conv2d_samepadding(256, 128, 3, padding='SAME'),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_2 = nn.Sequential(
            Conv2d_modified.Conv2d_samepadding(128, 64, 3, padding='SAME'),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )


        self.conv_3 = nn.Sequential(
            Conv2d_modified.Conv2d_samepadding(64, 32, 3, padding='SAME'),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv_4 = nn.Sequential(
            Conv2d_modified.Conv2d_samepadding(32, 1, 1, padding='SAME')
        )



    def forward(self,x):

        rf = self.Featex(x) #(batch, channel=256, H, W)
        feature = self.ann_nl_1(rf) #(batch, channel=256, H, W)
        feature = self.conv_1(feature) #(batch, channel=128, H, W)
        feature = self.conv_2(feature)
        feature = self.ann_nl_2(feature) #(batch, channel=64, H, W)
        feature = self.conv_3(feature)
        pred_out = self.conv_4(feature)

        return pred_out

def create_model(IMC_model_idx, freeze_featex, window_size_list=[7,15,31], OPTS_aspp = False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    type_idx = IMC_model_idx if IMC_model_idx < 4 else 2
    Featex = create_featex.Featex_vgg16_base(type_idx)
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
    model = ManTraNet(Featex, pool_size_list=window_size_list, is_dynamic_shape=True, apply_normalization=True, OPTS_aspp=OPTS_aspp)
    return model


def model_load_weights(weight_path, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_weights.load_weights(weight_path, model, True)    
    model.to(device)

    return model