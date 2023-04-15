_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712_mod.py',
    '../_base_/schedules/schedule_20e.py', '../_base_/default_runtime.py'
]
model = dict(
    pretrained='pretrained/pcpvt_small.pth',
    backbone=dict(
        type='pcpvt_small',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5))
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=6)
optimizer = dict(type='AdamW', lr=0.00005, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 19])
total_epochs = 20