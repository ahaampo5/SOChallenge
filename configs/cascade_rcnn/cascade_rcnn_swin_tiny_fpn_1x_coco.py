_base_ = [
    '../_base_/models/cascade_rcnn_swin_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

fp16 = dict(loss_scale=512.)