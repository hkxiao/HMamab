_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py',
]
model_wrapper_cfg=dict(
    type='MMDistributedDataParallel', find_unused_parameters=True)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained="../classification/vssm_tiny_0230_ckpt_epoch_262.pth",
        dims=96,
        depths=(2, 2, 5, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz", # v3_noz,
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.35,
        k_group= 4,
        imgsize=512,
        directions = [
            ['hilbert_a', 'hilbert_a_flip', 'hcurve_a' , 'hcurve_a_flip'],
            ['h', 'v', 'h_flip' , 'v_flip'],

            ['hilbert_a', 'hilbert_a_flip', 'hcurve_a' , 'hcurve_a_flip'],
            ['h', 'v', 'h_flip' , 'v_flip'],

            ['hilbert_a', 'hilbert_a_flip', 'hcurve_a' , 'hcurve_a_flip'],
            ['h', 'v', 'h_flip' , 'v_flip'],
            ['hilbert_a', 'hilbert_a_flip', 'hcurve_a' , 'hcurve_a_flip'],
            ['h', 'v', 'h_flip' , 'v_flip'],
            ['hilbert_a', 'hilbert_a_flip', 'hcurve_a' , 'hcurve_a_flip'],
            
            ['h', 'v', 'h_flip' , 'v_flip'],
            ['hilbert_a', 'hilbert_a_flip', 'hcurve_a' , 'hcurve_a_flip'],
        ],
        direction_aware = False,
        sc_attn = False,
        posembed=True
                
    ),
    
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=dict(
        #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)), default setting
    
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False,loss_weight=1.0), # 多种 loss 
        #    dict(type='DiceLoss', loss_name='loss_dice',use_sigmoid=False, loss_weight=1.0)
        ],
        #sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)
    ),
    
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        
        loss_decode=[
        dict(type='CrossEntropyLoss', use_sigmoid= False, loss_name='loss_ce', loss_weight=0.4),
        #dict(type='DiceLoss', use_sigmoid=False, loss_name='loss_dice', loss_weight=0.4)
        ]
        
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

train_dataloader = dict(batch_size=8) # as gpus=4



optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00003, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'FCNHead': dict(lr_mult=10.), ## 主干网络和解码头组件使用不同的学习率  
            'UPerHead': dict(lr_mult=10.) ## 主干网络和解码头组件使用不同的学习率             
        }))



param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=0.80,
        begin=1500,
        end=320000,
        by_epoch=False,
    )
]






# training schedule for 320k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=320000, val_interval=16000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=16000))





# dataset settings 加大尺度
dataset_type = 'ADE20KDataset'
data_root = '../../../data/ADEChallengeData2016'
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.25, 2.75),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='PackSegInputs')
]


img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator


