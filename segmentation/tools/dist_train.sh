CONFIG=configs/hssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_mlp_ratio_bad_samples_tuning.py
GPUS=8
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3} \
    --work-dir output/hcurve-hilbert-zigzag-absposem_bad_samples_tuning \
    --load_from output/hcurve-hilbert-zigzag-absposem_bad_samples_tuning/iter_1600.pth