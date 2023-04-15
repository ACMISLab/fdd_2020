_base_ = 'retinanet_pvtv2-b0_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                      'releases/download/v2/pvt_v2_b2.pth')),
    neck=dict(in_channels=[64, 128, 320, 512]))
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        times=3)
        )
