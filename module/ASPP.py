import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, num_classes):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(128)

        self.conv_3x3_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(128)

        self.conv_3x3_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(128)

        self.conv_3x3_3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(128)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(256, 128, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(128)

        self.conv_1x1_3 = nn.Conv2d(640, 128, kernel_size=1) # (640 = 5*128)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(128)

        self.conv_1x1_4 = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 256, h, w)) 
        feature_map_h = feature_map.size()[2] # (== h)
        feature_map_w = feature_map.size()[3] # (== w)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))   # (shape: (batch_size, 128, h, w))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 128, h, w))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 128, h, w))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 128, h, w))

        out_img = self.avg_pool(feature_map) # (shape: (batch_size, 256, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 128, 1, 1))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode='bilinear', align_corners = True) # (shape: (batch_size, 128, h, w))
                 #F.upsample has deprecated, turning to F.interpolate
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 640, h, w))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 128, h, w))
        out = self.conv_1x1_4(out) # (shape: (batch_size, num_classes=2, h, w))

        return out