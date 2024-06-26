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
        #pretrained="../classification/ckpt_epoch_288_.pth",
        # copied from classification/configs/vssm/vssm_tiny_224.yaml
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
        # directions = [
        #     ['h', 'v', 'h_flip' , 'v_flip'],
        #     ['h', 'v', 'h_flip' , 'v_flip'],

        #     ['h', 'v', 'h_flip' , 'v_flip'],
        #     ['h', 'v', 'h_flip' , 'v_flip'],

        #     ['h', 'v', 'h_flip' , 'v_flip'],
        #     ['h', 'v', 'h_flip' , 'v_flip'],
        #     ['h', 'v', 'h_flip' , 'v_flip'],
        #     ['h', 'v', 'h_flip' , 'v_flip'],
        #     ['h', 'v', 'h_flip' , 'v_flip'],
            
        #     ['h', 'v', 'h_flip' , 'v_flip'],
        #     ['h', 'v', 'h_flip' , 'v_flip'],
        # ],
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

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1))
