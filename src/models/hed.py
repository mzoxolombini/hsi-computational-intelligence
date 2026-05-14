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
        if pretrained:
            try:
                vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            except AttributeError:
                # Fallback for torch < 1.13
                vgg16 = models.vgg16(pretrained=True)  # noqa: TOR101
        else:
            try:
                vgg16 = models.vgg16(weights=None)
            except TypeError:
                vgg16 = models.vgg16(pretrained=False)  # noqa: TOR101
        
        # Extract feature layers
        self.block1 = vgg16.features[0:4]    # 2×(Conv+ReLU), output 64ch
        self.pool1 = vgg16.features[4]       # MaxPool
        self.block2 = vgg16.features[5:9]    # 2×(Conv+ReLU), output 128ch
        self.pool2 = vgg16.features[9]
        self.block3 = vgg16.features[10:16]  # 3×(Conv+ReLU), output 256ch
        self.pool3 = vgg16.features[16]
        self.block4 = vgg16.features[17:23]  # 3×(Conv+ReLU), output 512ch
        self.pool4 = vgg16.features[23]
        self.block5 = vgg16.features[24:30]  # 3×(Conv+ReLU), output 512ch
        self.pool5 = vgg16.features[30]
        
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
        h = self.block1(x)
        side1 = self.side1(h)
        side1 = F.interpolate(side1, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        h = self.pool1(h)
        h = self.block2(h)
        side2 = self.side2(h)
        side2 = F.interpolate(side2, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        h = self.pool2(h)
        h = self.block3(h)
        side3 = self.side3(h)
        side3 = F.interpolate(side3, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        h = self.pool3(h)
        h = self.block4(h)
        side4 = self.side4(h)
        side4 = F.interpolate(side4, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        h = self.pool4(h)
        h = self.block5(h)
        side5 = self.side5(h)
        side5 = F.interpolate(side5, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Fusion
        fuse = torch.cat([side1, side2, side3, side4, side5], dim=1)
        fuse = self.fuse(fuse)
        
        return [side1, side2, side3, side4, side5, fuse]
