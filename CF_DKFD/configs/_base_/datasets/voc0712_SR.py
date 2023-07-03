dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='VOCDataset',
            ann_file=['data/VOCdevkit/VOC2007/ImageSets/Main/trainval_SR_USM.txt'],
            img_prefix=['data/VOCdevkit/VOC2007/'],
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='Resize',
                    img_scale=[(480, 1000), (512, 1000), (544, 1000),
                               (576, 1000), (608, 1000), (640, 1000),
                               (672, 1000), (704, 1000), (736, 1000),
                               (768, 1000), (800, 1000)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ])),
    val=dict(
        type='VOCDataset',
        ann_file='data/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
        img_prefix='data/VOCdevkit/VOC2007/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(544, 1000), (640, 1000), (736, 1000), (800, 1000)],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='VOCDataset',
        ann_file='data/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
        img_prefix='data/VOCdevkit/VOC2007/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(544, 1000), (640, 1000), (736, 1000), (800, 1000)],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='mAP')