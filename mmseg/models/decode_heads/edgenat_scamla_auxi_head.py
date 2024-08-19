import torch
import torch.nn as nn

from mmseg.ops import Upsample

from ..builder import HEADS
from .decode_head import BaseDecodeHead

@HEADS.register_module()
class EdgeNAT_SCAMLA_AUXIHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(EdgeNAT_SCAMLA_AUXIHead, self).__init__(**kwargs)
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // 2, kernel_size=3, bias=False),
            nn.BatchNorm2d(self.in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels // 2, self.num_classes, kernel_size=1, bias=False),
        )
        
        self.up = Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self._transform_inputs(x)
        x = self.conv_out(x)
        x = self.up(x)
        edge = torch.sigmoid(x)
        return edge