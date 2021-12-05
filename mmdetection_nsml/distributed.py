import os


os.system(
'''
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=-29500 \
    $(dirname "$0")/mmdet.py 1 --launcher pytorch ${@:3}
'''
)