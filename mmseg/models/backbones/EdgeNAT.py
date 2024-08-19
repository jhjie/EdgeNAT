import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath
from mmcv.runner import load_checkpoint
from mmcv.runner import BaseModule

from . import layers

from mmseg.ops import Upsample
from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES

from natten import NeighborhoodAttention2D as NeighborhoodAttention


class ConvTokenizer(BaseModule):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                embed_dim // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(BaseModule):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(
            dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class Mlp(BaseModule):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NATLayer(BaseModule):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class NATBlock(BaseModule):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        kernel_size,
        dilations=None,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                NATLayer(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                )
                for i in range(depth)
            ]
        )

        self.downsample = (
            None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x, x
        return self.downsample(x), x


class EdgeNAT_MLA(BaseModule):
    def __init__(self, backbone_out_chs, backbone_indices, cm_bin_sizes, cm_out_ch, 
                 arm_type, resize_mode, fpn_Bu=False, align_corners=False):
        super(EdgeNAT_MLA, self).__init__()

        self.align_corners = align_corners
        self.fpn_Bu=fpn_Bu
        self.arm_type=arm_type
        self.cm_bin_sizes=cm_bin_sizes
        
        if self.cm_bin_sizes is not None:
            self.cm = ContextModule(backbone_out_chs[-1], cm_out_ch, cm_out_ch, cm_bin_sizes)
        if self.arm_type is not None:
            assert hasattr(layers, arm_type), \
                "Not support arm_type ({})".format(arm_type)
            arm_class = eval("layers." + arm_type)

            self.arm_list = nn.ModuleList()  # [..., arm8, arm16, arm32]
            for i in range(len(backbone_indices)):
                low_chs = backbone_out_chs[backbone_indices[i]]
                high_ch = cm_out_ch if i == (len(backbone_indices) - 1) else backbone_out_chs[backbone_indices[i + 1]]
                out_ch = backbone_out_chs[backbone_indices[i]]
                arm = arm_class(
                    low_chs, high_ch, out_ch, ksize=3, resize_mode=resize_mode)
                self.arm_list.append(arm)

        self.deconvs = nn.ModuleList()
        for i in range(len(backbone_indices) - 1):
            self.deconvs.append(
                nn.Sequential(
                    nn.Conv2d(backbone_out_chs[i+1], backbone_out_chs[i], kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(backbone_out_chs[i]),
                    nn.ReLU(inplace=True),
                    Upsample(scale_factor=2, mode=resize_mode, align_corners=align_corners)
                )
            )
        if self.arm_type is not None:
            self.out_convs = nn.ModuleList()
            for i in range(len(backbone_indices)):
                self.out_convs.append(
                    nn.Sequential(
                        nn.Conv2d(backbone_out_chs[i] * 3, backbone_out_chs[i], kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(backbone_out_chs[i]),
                        nn.ReLU(inplace=True),
                    )
                )
        else:
            self.out_convs = nn.ModuleList()
            for i in range(len(backbone_indices)):
                self.out_convs.append(
                    nn.Sequential(
                        nn.Conv2d(backbone_out_chs[i] * 2, backbone_out_chs[i], kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(backbone_out_chs[i]),
                        nn.ReLU(inplace=True),
                    )
                )

        if self.fpn_Bu:
            self.convs_Bu = nn.ModuleList()
            for i in range(len(backbone_indices)-1):
                self.convs_Bu.append(
                    nn.Sequential(
                        nn.Conv2d(backbone_out_chs[i], backbone_out_chs[i+1], kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(backbone_out_chs[i+1]),
                        nn.ReLU(inplace=True),
                    )
                )

            self.out_convs_Bu = nn.ModuleList()
            for i in range(len(backbone_indices)):
                self.out_convs_Bu.append(
                    nn.Sequential(
                        nn.Conv2d(backbone_out_chs[i] * 2, backbone_out_chs[i], kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(backbone_out_chs[i]),
                        nn.ReLU(inplace=True),
                    )
                )

    def forward(self, in_feat_list):
        if self.cm_bin_sizes is not None:
            cm_out = self.cm(in_feat_list[-1])
        else:
            cm_out = in_feat_list[-1]
        out_feat_list = []
        if self.arm_type is not None:
            high_feat = cm_out
            for i in reversed(range(len(in_feat_list))):
                low_feat = in_feat_list[i]
                arm = self.arm_list[i]
                arm_feat = arm(low_feat, high_feat)
                if low_feat.shape == high_feat.shape:
                    high_feat = torch.cat((arm_feat, high_feat, low_feat), dim=1)
                else:
                    down_feat = self.deconvs[i](high_feat)
                    high_feat = torch.cat((arm_feat, down_feat, low_feat), dim=1)
                high_feat = self.out_convs[i](high_feat)
                out_feat_list.insert(0, high_feat)
        else:
            for i in reversed(range(len(in_feat_list))):
                if i == len(in_feat_list)-1:
                    if self.cm_bin_sizes is not None:
                        feat = torch.cat((in_feat_list[i], cm_out), dim=1)
                        feat = self.out_convs[i](feat)
                    else:
                        feat = in_feat_list[i]
                else:
                    Td_feat = self.deconvs[i](feat)
                    feat = torch.cat((in_feat_list[i], Td_feat), dim=1)
                    feat = self.out_convs[i](feat)
                out_feat_list.insert(0, feat)

        if self.fpn_Bu:
            for i in range(len(in_feat_list)):
                if i > 0:
                    Bu_feat = self.convs_Bu[i-1](feat)
                    feat = torch.cat((in_feat_list[i], Bu_feat), dim=1)
                    feat = self.out_convs_Bu[i](feat)
                else:
                    feat = in_feat_list[i]
                out_feat_list.append(feat)
        return out_feat_list

class ContextModule(BaseModule):
    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 bin_sizes,
                 align_corners=False):
        super(ContextModule, self).__init__()

        self.stages = nn.ModuleList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])

        self.conv_out = nn.Sequential(
            nn.Conv2d(
                in_channels=inter_channels * len(bin_sizes) + in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=size)
        conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return nn.Sequential(prior, conv)

    def forward(self, input):
        out = None
        input_shape = input.shape[2:]
        r = input

        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(
                x,
                input_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            if out is None:
                out = x
            else:
                out = torch.cat((out, x), dim=1)

        out = torch.cat((out, r), dim=1)
        out = self.conv_out(out)
        return out

class EDHead(BaseModule):
    def __init__(self, idxs, in_chan, backbone_out_chs, resize_mode, align_corners=False):
        super(EDHead, self).__init__()

        self.align_corners = align_corners

        self.convs = nn.ModuleList()
        for idx in reversed(range(idxs)):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_chan, backbone_out_chs[idx], kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(backbone_out_chs[idx]),
                    nn.ReLU(inplace=True),
                )
            )
            in_chan = backbone_out_chs[idx]

        self.up = Upsample(scale_factor=2 ** idxs, mode=resize_mode, align_corners=align_corners)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.up(x)
        return x


@BACKBONES.register_module()
class EdgeNAT(BaseModule):
    def __init__(
        self,
        embed_dim,
        mlp_ratio,
        depths,
        num_heads,
        drop_path_rate=0.2,
        in_chans=3,
        kernel_size=7,
        dilations=None,
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        frozen_stages=-1,
        pretrained=None,
        layer_scale=None,
        num_classes=1,
        backbone_indices=[0, 1, 2, 3],
        arm_type='FFM_SCAtten',
        cm_bin_sizes=[1, 2, 3, 6],
        cm_out_ch=512,
        fpn_Bu=False,
        resize_mode='bilinear',
        **kwargs,
    ):
        super(EdgeNAT, self).__init__()
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.features_chan = [int(embed_dim * 2**i) for i in range(self.num_levels)]
        self.mlp_ratio = mlp_ratio

        self.patch_embed = ConvTokenizer(
            in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.backbone_indices = backbone_indices
    
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size,
                dilations=None if dilations is None else dilations[i],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
            )
            self.levels.append(level)

        # add a norm layer for each output
        self.out_indices = out_indices
        for i_layer in self.out_indices:
            layer = norm_layer(self.features_chan[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self.frozen_stages = frozen_stages
        if pretrained is not None:
            self.initWeights(pretrained)

        self.fedter_head = EdgeNAT_MLA(self.features_chan, 
                                      self.backbone_indices, 
                                      cm_bin_sizes, 
                                      cm_out_ch, 
                                      arm_type,
                                      resize_mode,
                                      fpn_Bu=fpn_Bu,
                                      )

        arm_out_chs = [self.features_chan[i] for i in self.backbone_indices]
        if fpn_Bu:
            arm_out_chs = arm_out_chs + arm_out_chs
        self.ed_heads = nn.ModuleList()
        for idx, in_ch in enumerate(arm_out_chs):
            idx = idx % len(self.backbone_indices)
            self.ed_heads.append(EDHead(self.backbone_indices[idx], in_ch, self.features_chan, resize_mode))

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            for i in range(0, self.frozen_stages - 1):
                m = self.network[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(EdgeNAT, self).train(mode)
        self._freeze_stages()

    def initWeights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError("pretrained must be a str or None")

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, level in enumerate(self.levels):
            x, xo = level(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(xo)
                outs.append(x_out.permute(0, 3, 1, 2).contiguous())
        return outs

    def forward(self, x):
        x = self.forward_embeddings(x)
        outs =  self.forward_tokens(x)
        feats_selected = [outs[i] for i in self.backbone_indices]

        feats = self.fedter_head(feats_selected)

        logit_list = []
        for x, ed_head in zip(feats, self.ed_heads):
            x = ed_head(x)
            logit_list.append(x)

        return tuple(logit_list)
