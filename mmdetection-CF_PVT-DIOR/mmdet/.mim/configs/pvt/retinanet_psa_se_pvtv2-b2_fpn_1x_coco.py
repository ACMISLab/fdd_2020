_base_ = 'retinanet_pvtv2-b2_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        _delete_=True,
        type='PyramidVisionTransformerV2PSA_cbam2',
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                      'releases/download/v2/pvt_v2_b2.pth')),
    neck=dict(in_channels=[64, 128, 320, 512]))
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)