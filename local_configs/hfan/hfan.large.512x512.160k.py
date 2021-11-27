# dataset settings
dataset_type = 'VOSDataset'
data_root_DAVIS = '/hfan/data/DAVIS2SEG/'
data_root_YT = '/hfan/data/YouTube2SEG'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImagesFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortionMultiImages'),
    dict(type='NormalizeMultiImages', **img_norm_cfg),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImagesFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            # dict(type='AlignedResize', keep_ratio=True, size_divisor=32),
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='NormalizeMultiImages', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

YT_train = dict(
    type=dataset_type,
    data_root=data_root_YT,
    img1_dir='frame/train',
    img2_dir='flow/train',
    ann_dir='mask/train',
    pipeline=train_pipeline)

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=[YT_train],
    val=dict(
        type=dataset_type,
        data_root=data_root_DAVIS,
        img1_dir='frame/val',
        img2_dir='flow/val',
        ann_dir='mask/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root_DAVIS,
        img1_dir='frame/val',
        img2_dir='flow/val',
        ann_dir='mask/val',
        pipeline=test_pipeline))

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='/hfan/checkpoints/mit_b3.pth',
    backbone=dict(
        type='HFANVOS',
        ori_type='mit_b3',
        style='pytorch'),
    decode_head=dict(
        type='HFANVOS_Head',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        select_method='hfan',     # hfan
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    # test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(271, 271))
)
# optimizer
optimizer = dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=3000,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16000, metric=['VOS'])
optimizer_config = dict()

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook', interval=10)
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

