_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712_aug_SR_2x.py',
    '../_base_/schedules/schedule_20e.py', '../_base_/default_runtime.py'
]
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[12, 16])
# runner = dict(type='EpochBasedRunner', max_epochs=17)