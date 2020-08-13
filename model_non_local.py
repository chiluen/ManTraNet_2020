import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from module import Conv2d_modified, ConvLSTM, create_featex, NestedWindow, GlobalStd2D, ASPP
#from module.nonlocal_block.non_local_embedded_gaussian import NONLocalBlock2D
from module.nonlocal_block.non_local_dot_product import NONLocalBlock2D
#from module.nonlocal_block.non_local_concatenation import NONLocalBlock2D
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
        first_channel = 32 #128太大
        second_channel = 16

        self.conv_1 = nn.Sequential( 
            Conv2d_modified.Conv2d_samepadding(256, first_channel, 3, padding='SAME'),
            nn.BatchNorm2d(first_channel),
            nn.ReLU()
        )
        self.nl_1 = NONLocalBlock2D(in_channels=first_channel)

        self.conv_2 = nn.Sequential(
            Conv2d_modified.Conv2d_samepadding(first_channel, second_channel, 3, padding='SAME'),
            nn.BatchNorm2d(second_channel),
            nn.ReLU()
        )
        self.nl_2 = NONLocalBlock2D(in_channels=second_channel)

        self.conv_3 = nn.Sequential(
            Conv2d_modified.Conv2d_samepadding(second_channel, 1, 3, padding='SAME'),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )



        """
        #layers
        self.outlierTrans = Conv2d_modified.Conv2d_samepadding(256, 64, 1, padding='SAME')  #在optimizer那邊要做apply_normalization
        self.pred = Conv2d_modified.Conv2d_samepadding(8, 1, 7, padding="SAME", bias=True)  
        self.bnorm = nn.BatchNorm2d(64, affine=False, eps=1e-3)
        self.nestedAvgFeatex = NestedWindow.NestedWindowAverageFeatExtrator(window_size_list= self.pool_size_list, 
                                                                               output_mode='5d',
                                                                               minus_original=True) 
        self.glbStd = GlobalStd2D.GlobalStd2D(64)   #input: number of features
        self.cLSTM = ConvLSTM.ConvLSTM(input_dim = 64, hidden_dim = 8, kernel_size = (7, 7), num_layers = 1, batch_first = True, bias = True, return_all_layers = False)
        self.sigmoid = nn.Sigmoid()
        self.featex = None
        self.clstm = None
        #self.aspp = ASPP.ASPP(2)
        self.OPTS_aspp = OPTS_aspp
        """
    def forward(self,x):
        rf = self.Featex(x) #(batch, channel=256, H, W)
        #ipdb.set_trace()
        #Non-local network

        feature_1 = self.conv_1(rf)
        nl_feature_1 = self.nl_1(feature_1)

        del feature_1
        feature_2 = self.conv_2(nl_feature_1)
        #nl_feature_2 = self.nl_2(feature_2)
        #pred_out = self.conv_3(nl_feature_2)

        pred_out = self.conv_3(feature_2)

        #pred_out = self.sigmoid(pred_out)

        """
        #aspp_temp = rf
        self.featex = rf
        rf = self.outlierTrans(rf) 
        bf = self.bnorm(rf) #(batch, channel=64, H, W)
        devf5d = self.nestedAvgFeatex(bf) #(batch, 4, channel=64, H, W)
        if self.apply_normalization:
            sigma = self.glbStd(bf) #(batch, channel, H, W)   
            sigma5d = torch.unsqueeze(sigma, 1) 
            devf5d = torch.abs(devf5d / sigma5d) 

        # Convert back to 4d
        _, last_states = self.cLSTM(devf5d)  
        devf = last_states[0][0] #(batch, channel = 8, H, W)
        self.clstm = devf.detach().cpu().numpy()
        pred_out = self.pred(devf) #(batch, channel = 1, H, W)
        pred_out = self.sigmoid(pred_out)
        """


        return pred_out

        #print fm
    def feature_map(self, fm = 'featex'):
        if fm == 'featex':
            return self.featex
        elif fm == 'clstm':
            return self.clstm

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