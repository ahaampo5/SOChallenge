_base_ = [
    '../_base_/models/cascade_rcnn_swin_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
<<<<<<< HEAD
=======

fp16 = dict(loss_scale=512.)
>>>>>>> 8e8b0258373baf9d9a76420023d861cd450cd7e4
