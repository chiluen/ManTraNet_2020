import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from module import Conv2d_modified, ConvLSTM, create_featex, NestedWindow, GlobalStd2D, ASPP, AFNB
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

        self.ann_nl = AFNB.AFNB(low_in_channels= first_channel, high_in_channels=256, out_channels=256, 
                           key_channels=first_channel, value_channels=first_channel, dropout=0.05)

        self.ann_nl_2 = AFNB.AFNB(low_in_channels= second_channel, high_in_channels=first_channel, out_channels=first_channel, 
                             key_channels=second_channel, value_channels=second_channel, dropout=0.05)


        self.conv_1 = nn.Sequential( 
            Conv2d_modified.Conv2d_samepadding(256, first_channel, 3, padding='SAME'),
            nn.BatchNorm2d(first_channel),
            nn.ReLU()
        )

        self.conv_2 = nn.Sequential(
            Conv2d_modified.Conv2d_samepadding(256, first_channel, 3, padding='SAME'),
            nn.BatchNorm2d(first_channel),
            nn.ReLU()
        )


        self.conv_3 = nn.Sequential(
            Conv2d_modified.Conv2d_samepadding(first_channel, second_channel, 3, padding='SAME'),
            nn.BatchNorm2d(second_channel),
            nn.ReLU()
        )

        self.conv_4 = nn.Sequential(
            Conv2d_modified.Conv2d_samepadding(first_channel, 64, 1, padding='SAME'),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv_5 = nn.Sequential(
            Conv2d_modified.Conv2d_samepadding(64, 1, 1, padding='SAME'),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )




    def forward(self,x):

        #-------- 256(High) -> 128(Low) -> 256(After non_local) --------#
        rf = self.Featex(x) #(batch, channel=256, H, W)
        feature_1 = self.conv_1(rf) #(batch, 128,H,W)
        nl_feature_1 = self.ann_nl(feature_1, rf) #(batch, 256,H,W)
        
        #-------- 128(High) -> 64(Low) -> 128(After non_local) --------#
        feature_2 = self.conv_2(nl_feature_1) #(batch, 128,H,W)
        feature_3 = self.conv_3(feature_2) #(batch, 64,H,W)
        nl_feature_2 = self.ann_nl_2(feature_3, feature_2) #(batch, 128,H,W)
        
        #-------- 128 -> 64 --------#
        pred_out = self.conv_4(nl_feature_2) #(batch, 64, H, W)
        pred_out = self.conv_5(pred_out) #(batch, 1, H, W)


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