# src/models/hed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class HolisticallyNestedEdgeDetection(nn.Module):
    """
    Holistically-Nested Edge Detection (HED) Network
    Based on Xie & Tu (2015)
    """
    
    def __init__(self, pretrained=True):
        super(HolisticallyNestedEdgeDetection, self).__init__()
        
        # Load VGG16 as backbone
        vgg16 = models.vgg16(pretrained=pretrained)
        
        # Extract feature layers
        self.conv1_1 = vgg16.features[0:2]   # Conv + ReLU
        self.conv1_2 = vgg16.features[2:5]   # Conv + ReLU + MaxPool
        self.conv2_1 = vgg16.features[5:7]
        self.conv2_2 = vgg16.features[7:10]
        self.conv3_1 = vgg16.features[10:12]
        self.conv3_2 = vgg16.features[12:14]
        self.conv3_3 = vgg16.features[14:17]
        self.conv4_1 = vgg16.features[17:19]
        self.conv4_2 = vgg16.features[19:21]
        self.conv4_3 = vgg16.features[21:24]
        self.conv5_1 = vgg16.features[24:26]
        self.conv5_2 = vgg16.features[26:28]
        self.conv5_3 = vgg16.features[28:31]
        
        # Side output layers
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(128, 1, kernel_size=1)
        self.side3 = nn.Conv2d(256, 1, kernel_size=1)
        self.side4 = nn.Conv2d(512, 1, kernel_size=1)
        self.side5 = nn.Conv2d(512, 1, kernel_size=1)
        
        # Fusion layer
        self.fuse = nn.Conv2d(5, 1, kernel_size=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Forward pass through VGG layers
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        side1 = self.side1(h)
        side1 = F.interpolate(side1, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        h = self.conv2_1(h)
        h = self.conv2_2(h)
        side2 = self.side2(h)
        side2 = F.interpolate(side2, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)
        side3 = self.side3(h)
        side3 = F.interpolate(side3, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_3(h)
        side4 = self.side4(h)
        side4 = F.interpolate(side4, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)
        side5 = self.side5(h)
        side5 = F.interpolate(side5, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Fusion
        fuse = torch.cat([side1, side2, side3, side4, side5], dim=1)
        fuse = self.fuse(fuse)
        
        return [side1, side2, side3, side4, side5, fuse]