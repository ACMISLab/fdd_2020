#_base_ = './faster_rcnn_pcpvt_s_fpn_1x_coco_pvt_setting.py'
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712_aug.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    pretrained='pretrained/alt_gvt_small.pth',
    backbone=dict(
        type='alt_gvt_small',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=5))
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=6)

