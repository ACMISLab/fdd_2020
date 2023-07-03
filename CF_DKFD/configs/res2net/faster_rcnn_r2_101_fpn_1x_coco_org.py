_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712_mod.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://res2net101_v1d_26w_4s')))
