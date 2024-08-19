# 此处需要进一步修改模型

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import fusion_helper as helper

class FFM(nn.Module):
    """
    The base of Unified Attention Fusion Module.
    Args:
        l_ch (int): The channel of l tensor, which is the low level feature.
        h_ch (int): The channel of h tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for l tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling h tensor. Default: bilinear.
    """

    def __init__(self, l_ch, h_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super(FFM, self).__init__()

        self.conv_l = nn.Sequential(
            nn.Conv2d(l_ch, h_ch, kernel_size=ksize, padding=ksize // 2, bias=False),
            nn.BatchNorm2d(h_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(h_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.resize_mode = resize_mode

    def check(self, l, h):
        assert l.dim() == 4 and h.dim() == 4
        l_h, l_w = l.shape[2:]
        h_h, h_w = h.shape[2:]
        assert l_h >= h_h and l_w >= h_w

    def prepare(self, l, h):
        l = self.prepare_l(l, h)
        h = self.prepare_h(l, h)
        return l, h

    def prepare_l(self, l, h):
        l = self.conv_l(l)
        return l

    def prepare_h(self, l, h):
        h_up = F.interpolate(h, size=l.shape[2:], mode=self.resize_mode)
        return h_up

    def fuse(self, l, h):
        out = l + h
        out = self.conv_out(out)
        return out

    def forward(self, l, h):
        """
        Args:
            l (Tensor): The low level feature.
            h (Tensor): The high level feature.
        """
        self.check(l, h)
        l, h = self.prepare(l, h)
        out = self.fuse(l, h)
        return out


class FFM_ChAtten(FFM):
    """
    The UAFM with channel attention, which uses mean and max values.
    Args:
        l_ch (int): The channel of l tensor, which is the low level feature.
        h_ch (int): The channel of h tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for l tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling h tensor. Default: bilinear.
    """

    def __init__(self, l_ch, h_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(l_ch, h_ch, out_ch, ksize, resize_mode)

        self.conv_lh_atten = nn.Sequential(
            nn.Conv2d(4 * h_ch, h_ch // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(h_ch // 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(h_ch // 2, h_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(h_ch),
        )

    def fuse(self, l, h):
        """
        Args:
            l (Tensor): The low level feature.
            h (Tensor): The high level feature.
        """
        atten = helper.avg_max_reduce_hw([l, h])
        atten = torch.sigmoid(self.conv_lh_atten(atten))

        out = l * atten + h * (1 - atten)
        out = self.conv_out(out)
        return out


class FFM_SpAtten(FFM):
    """
    The UAFM with spatial attention, which uses mean and max values.
    Args:
        l_ch (int): The channel of l tensor, which is the low level feature.
        h_ch (int): The channel of h tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for l tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling h tensor. Default: bilinear.
    """

    def __init__(self, l_ch, h_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(l_ch, h_ch, out_ch, ksize, resize_mode)

        self.conv_lh_atten = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
        )
        self._scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def fuse(self, l, h):
        """
        Args:
            l (Tensor): The low level feature.
            h (Tensor): The high level feature.
        """
        atten = helper.mean_max_reduce_channel([l, h])
        atten = torch.sigmoid(self.conv_lh_atten(atten))

        out = l * atten + h * (self._scale - atten)
        out = self.conv_out(out)
        return out

class FFM_SCAtten(FFM):
    """
    The UAFM with spatial and channel attention, which uses mean and max values.
    Args:
        l_ch (int): The channel of l tensor, which is the low level feature.
        h_ch (int): The channel of h tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for l tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling h tensor. Default: bilinear.
    """

    def __init__(self, l_ch, h_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(l_ch, h_ch, out_ch, ksize, resize_mode)

        self.conv_lh_s_atten = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
        )
        self._scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.conv_lh_c_atten = nn.Sequential(
            nn.Conv2d(4 * h_ch, h_ch // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(h_ch // 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(h_ch // 2, h_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(h_ch),
        )

        self.conv_sc_out = nn.Sequential(
            nn.Conv2d(h_ch * 2, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def fuse(self, l, h):
        """
        Args:
            l (Tensor): The low level feature.
            h (Tensor): The high level feature.
        """

        atten_s = helper.mean_max_reduce_channel([l, h])
        atten_s = torch.sigmoid(self.conv_lh_s_atten(atten_s))

        out_s = l * atten_s + h * (self._scale - atten_s)

        atten_c = helper.avg_max_reduce_hw([l, h])
        atten_c = torch.sigmoid(self.conv_lh_c_atten(atten_c))

        out_c = l * atten_c + h * (1 - atten_c)

        out = torch.cat((out_s, out_c), dim=1)
        out = self.conv_sc_out(out)

        return out