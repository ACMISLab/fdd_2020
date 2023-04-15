_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712_aug_SR.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# custom hooks
custom_hooks = [dict(type='SetEpochInfoHook')]