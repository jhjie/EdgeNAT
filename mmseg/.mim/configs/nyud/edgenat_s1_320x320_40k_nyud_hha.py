_base_ = [
    '../_base_/models/fedter.py', '../_base_/datasets/nyud_hha.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(
    backbone=dict(
        type='EdgeNAT',
        embed_dim=64,
        mlp_ratio=3.0,
        depths=[3, 4, 18, 5],
        num_heads=[2, 4, 8, 16],
        drop_path_rate=0.3,
        kernel_size=7,
        dilations=[[1, 16, 1], [1, 4, 1, 8], [1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4], [1, 2, 1, 2, 1]],
        backbone_indices=[0, 1, 2, 3],
        arm_type='FFM_SCAtten',
        cm_bin_sizes=None,
        cm_out_ch=512,
        fpn_Bu=False,
        resize_mode='bilinear',
        pretrained='pretrained/dinat/dinat_tiny_in1k_224.pth',
    ),
    decode_head=dict(
        in_channels=64,
        channels=4,
        num_classes=1,
    ),
    auxiliary_head=[
        dict(
            type='EdgeNAT_SCAMLA_AUXIHead',
            in_channels=64,
            channels=1,
            in_index=0,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4
            )
        ),
        dict(
            type='EdgeNAT_SCAMLA_AUXIHead',
            in_channels=64,
            channels=1,
            in_index=1,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4
            )
        ),
        dict(
            type='EdgeNAT_SCAMLA_AUXIHead',
            in_channels=64,
            channels=1,
            in_index=2,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4
            )
        ),
        dict(
            type='EdgeNAT_SCAMLA_AUXIHead',
            in_channels=64,
            channels=1,
            in_index=3,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='HEDLoss', use_sigmoid=True, loss_weight=0.4
            )
        ),
    ])

# AdamW optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={
                     'rpb': dict(decay_mult=0.),
                     'norm': dict(decay_mult=0.),
                 }),)

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# test_cfg = dict(mode='slide', crop_size=(320, 320), stride=(280, 280))
# By default, models are trained on 1 GPUs with 8 images per GPU
data = dict(samples_per_gpu=4)

find_unused_parameters = True