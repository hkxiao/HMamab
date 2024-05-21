python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="127.0.0.1" --master_port=29501 main.py \
    --cfg configs/hssm/hmambav2_tiny_224_hilbert.yaml  \
    --batch-size 128 --data-path ../../../data/imagenet1k --output output \
    --pretrained pretrained/ckpt_epoch_288_.pth