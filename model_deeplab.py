import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from module import Conv2d_modified, ConvLSTM, create_featex, NestedWindow, GlobalStd2D, ASPP
import load_weights
import argparse
from module.unet.__init__ import *
import ipdb

##替代 create_manTraNet_model功能
class ManTraNet(nn.Module):
    def __init__(self, Featex, pool_size_list=[7,15,31], is_dynamic_shape=True, apply_normalization=True, mid_channel = 3):
        super().__init__()
        self.num_classes = 1
        self.Featex = Featex
        self.aspp = ASPP.ASPP(256)

        self.conv1 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn1  = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1))

    def forward(self,x):
        rf = self.Featex(x) #(batch, 256, H, W)
        pp = self.aspp(rf) #(batch, 256, H, W)
        #ipdb.set_trace()
        rf = self.conv1(rf) #(batch, 48, H, W) for 1x1 conv
        rf = self.relu(self.bn1(rf))

        pp = F.interpolate(pp, size=rf.size()[2:], mode='bilinear', align_corners=True) #這一部沒用
        block = torch.cat((pp, rf), dim=1) #(batch, 304, H, W)
        
        block = self.last_conv(block) #(batch, 1, H, W)
        #ipdb.set_trace()
        pred_out = F.interpolate(block, size=x.size()[2:], mode='bilinear', align_corners=True) #這一部沒用

        pred_out = torch.sigmoid(pred_out)
        return pred_out

def create_model(IMC_model_idx, freeze_featex, window_size_list=[7,15,31]):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    type_idx = IMC_model_idx if IMC_model_idx < 4 else 2
    Featex = create_featex.Featex_vgg16_base(type_idx)
    if freeze_featex:
        print("INFO: freeze feature extraction part, trainable=False")
        for p in Featex.parameters():
            p.requires_grad = False
    else:
        print ("INFO: unfreeze feature extraction part, trainable=True")
    """
    if len(window_size_list) == 4:
        for ly in Featex.layers[:5]:
            ly.trainable = False  ##也沒有這個選項
            print("INFO: freeze", ly.name)
    """
    model = ManTraNet(Featex, pool_size_list=window_size_list, is_dynamic_shape=True, apply_normalization=True)
    return model


def model_load_weights(weight_path, model, Featex_only=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_weights.load_weights(weight_path, model, Featex_only)    
    model.to(device)

    return model