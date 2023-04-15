_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712_mod.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    pretrained='pretrained/alt_gvt_small.pth',
    backbone=dict(
        type='alt_gvt_small',
        style='pytorch'),
    neck=dict(
        in_channels=[64, 128, 256, 512],
        out_channels=256))

