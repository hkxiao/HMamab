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
            ['h', 'v', 'hilbert_a' , 'hcurve_a'],
            ['h_flip', 'v_flip', 'hilbert_a_flip' , 'hcurve_a_flip'],
            
            ['h', 'v', 'hilbert_a' , 'hcurve_a'],
            ['h_flip', 'v_flip', 'hilbert_a_flip' , 'hcurve_a_flip'],

            ['h', 'v', 'hilbert_a' , 'hcurve_a'],
            ['h_flip', 'v_flip', 'hilbert_a_flip' , 'hcurve_a_flip'],
            ['h', 'v', 'hilbert_a' , 'hcurve_a'],
            ['h_flip', 'v_flip', 'hilbert_a_flip' , 'hcurve_a_flip'],
            ['h', 'v', 'hilbert_a' , 'hcurve_a'],
            
            ['h_flip', 'v_flip', 'hilbert_a_flip' , 'hcurve_a_flip'],            
            ['h', 'v', 'hilbert_a' , 'hcurve_a'],    
        ],
        direction_aware = True,
        sc_attn = True,
        posembed=True
                
    ),)

train_dataloader = dict(batch_size=4) # as gpus=4



