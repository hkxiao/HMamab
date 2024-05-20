CONFIG=configs/hssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_mlp_ratio.py
#CHECKPOINT=work_dirs_v0/upernet_vssm_4xb4-160k_ade20k-512x512_tiny/iter_160000.pth
CHECKPOINT=official_pretrained/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_iter_160000.pth
#CHECKPOINT=upernet_vssm_4xb4-160k_ade20k-512x512_tiny_iter_160000.pth
GPUS=8
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --work-dir official \
    --launcher pytorch \
    ${@:4}
