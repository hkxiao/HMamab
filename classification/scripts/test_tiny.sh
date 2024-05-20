python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 main.py \
    --cfg configs/vssm/vmambav2_tiny_224.yaml  \
    --batch-size 2 --data-path ../../../data/imagenet1k --output tmp \
    --pretrained pretrained/vssm_tiny_0230_ckpt_epoch_262.pth