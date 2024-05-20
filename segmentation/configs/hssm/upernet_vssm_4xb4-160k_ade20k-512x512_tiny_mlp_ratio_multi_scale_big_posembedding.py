
crop_size = [(512, 640) ,(512, 512), (640, 512), (512, 768), (768, 512)]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='PackSegInputs')
]

_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
model_wrapper_cfg=dict(
    type='MMDistributedDataParallel', find_unused_parameters=True)
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
        drop_path_rate=0.3,
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
                
    ),)

train_dataloader = dict(batch_size=4) # as gpus=4

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00003, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=0.75,
        begin=1500,
        end=320000,
        by_epoch=False,
    )
]

# training schedule for 160k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=320000, val_interval=16000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=16000))


