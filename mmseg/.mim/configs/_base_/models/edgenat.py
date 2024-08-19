# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='EdgeNAT',
        embed_dim=192,
        mlp_ratio=3.0,
        depths=[3, 4, 18, 5],
        num_heads=[6, 12, 24, 48],
        drop_path_rate=0.3,
        in_chans=3,
        kernel_size=7,
        dilations=[[1, 2, 1], [1, 2, 1, 5], [1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1, 9, 1, 10], [1, 10, 1, 20, 1]],
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        in_patch_size=4,
        frozen_stages=-1,
        numclasses=1,
        backbone_indices=[0, 1, 2, 3],
        arm_type='FFM_SCAtten',
        cm_bin_sizes=[1, 2, 3, 6],
        cm_out_ch=1536,
        resize_mode='bilinear',
    ),
    decode_head=dict(
        type='EdgeNAT_SCAMLAHead',
        in_channels=512,
        channels=4,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='HEDLoss', use_sigmoid=True, loss_weight=1.0)))
# model training and testing settings
train_cfg=dict()
test_cfg = dict(mode='whole')

