dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='VOCDataset',
            ann_file=['data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'],
            img_prefix=['data/VOCdevkit/VOC2007/'],
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='AutoAugment',
                    policies=[
                        [
                            {
                                'type': 'Resize',
                                'img_scale': [(480, 1000), (512, 1000), (544, 1000),
                                              (576, 1000), (608, 1000), (640, 1000),
                                              (672, 1000), (704, 1000), (736, 1000),
                                              (768, 1000), (800, 1000)],
                                'multiscale_mode': 'value',
                                'keep_ratio': True
                            }
                        ],
                        [
                            {
                                'type': 'Resize',
                                'img_scale': [(500, 1000), (600, 1000), (700, 1000)],
                                'multiscale_mode': 'value',
                                'keep_ratio': True
                            },
                            {
                                'type': 'RandomCrop',
                                'crop_type': 'absolute_range',
                                'crop_size': (400, 600),
                                'allow_negative_crop': False
                            },
                            {
                                'type': 'Resize',
                                'img_scale': [(480, 1000), (512, 1000),
                                              (544, 1000), (576, 1000),
                                              (608, 1000), (640, 1000),
                                              (672, 1000), (704, 1000),
                                              (736, 1000), (768, 1000),
                                              (800, 1000)],
                                'multiscale_mode': 'value',
                                'override': True,
                                'keep_ratio': True
                            }
                        ],
                        [
                            {
                                'type': 'Resize',
                                'img_scale': [(544, 1000), (640, 1000), (736, 1000)],
                                'multiscale_mode': 'value',
                                'keep_ratio': True
                            },
                            {
                                'type': 'MinIoURandomCrop',
                                'min_ious': (0.1, 0.3, 0.5, 0.7),
                            },
                            {
                                'type': 'Resize',
                                'img_scale': [(480, 1000), (512, 1000),
                                              (544, 1000), (576, 1000),
                                              (608, 1000), (640, 1000),
                                              (672, 1000), (704, 1000),
                                              (736, 1000), (768, 1000),
                                              (800, 1000)],
                                'multiscale_mode': 'value',
                                'override': True,
                                'keep_ratio': True
                            }
                        ],
                        [
                            {
                                'type': 'Resize',
                                'img_scale': [(480, 1000), (512, 1000), (544, 1000),
                                              (576, 1000), (608, 1000), (640, 1000),
                                              (672, 1000), (704, 1000), (736, 1000),
                                              (768, 1000), (800, 1000)],
                                'multiscale_mode': 'value',
                                'keep_ratio': True
                            },
                            {
                                'type': 'PhotoMetricDistortion'
                            }
                        ],
                        [
                            {
                                'type': 'Resize',
                                'img_scale': [(480, 1000), (512, 1000), (544, 1000),
                                              (576, 1000), (608, 1000), (640, 1000),
                                              (672, 1000), (704, 1000), (736, 1000),
                                              (768, 1000), (800, 1000)],
                                'multiscale_mode': 'value',
                                'keep_ratio': True
                            },
                            {
                                'type': 'Albu',
                                'transforms': albu_train_transforms,
                                'bbox_params': dict(
                                    type='BboxParams',
                                    format='pascal_voc',
                                    label_fields=['gt_labels'],
                                    min_visibility=0.0,
                                    filter_lost_elements=True),
                                'keymap': {
                                    'img': 'image',
                                    'gt_bboxes': 'bboxes'},
                                'update_pad_shape': False,
                                'skip_img_without_anno': True
                            }
                        ],
                        [
                            {
                                'type': 'Resize',
                                'img_scale': [(480, 1000), (512, 1000), (544, 1000),
                                              (576, 1000), (608, 1000), (640, 1000),
                                              (672, 1000), (704, 1000), (736, 1000),
                                              (768, 1000), (800, 1000)],
                                'multiscale_mode': 'value',
                                'keep_ratio': True
                            },
                            {
                                'type': 'RandomShift'
                            }
                        ],
                        [
                            {
                                'type': 'Resize',
                                'img_scale': [(480, 1000), (512, 1000), (544, 1000),
                                              (576, 1000), (608, 1000), (640, 1000),
                                              (672, 1000), (704, 1000), (736, 1000),
                                              (768, 1000), (800, 1000)],
                                'multiscale_mode': 'value',
                                'keep_ratio': True
                            },
                            {
                                'type': 'RandomAffine'
                            }
                        ],
                        [
                            {
                                'type': 'Resize',
                                'img_scale': [(480, 1000), (512, 1000), (544, 1000),
                                              (576, 1000), (608, 1000), (640, 1000),
                                              (672, 1000), (704, 1000), (736, 1000),
                                              (768, 1000), (800, 1000)],
                                'multiscale_mode': 'value',
                                'keep_ratio': True
                            },
                            {
                                'type': 'CutOut',
                                'n_holes': 3,
                                'cutout_shape': [(10, 20), (15, 30), (20, 40)],
                            }
                        ],
                        [
                            {
                                'type': 'Resize',
                                'img_scale': [(480, 1000), (512, 1000), (544, 1000),
                                              (576, 1000), (608, 1000), (640, 1000),
                                              (672, 1000), (704, 1000), (736, 1000),
                                              (768, 1000), (800, 1000)],
                                'multiscale_mode': 'value',
                                'keep_ratio': True
                            },
                            {
                                'type': 'ColorTransform',
                                'prob': 0.8,
                                'level': 4,
                            },
                            {
                                'type': 'Translate',
                                'prob': 1.0,
                                'level': 6,
                                'direction': 'vertical',
                            },
                            {
                                'type': 'Rotate',
                                'prob': 0.6,
                                'level': 6,
                            }
                        ],
                        [
                            {
                                'type': 'Resize',
                                'img_scale': [(480, 1000), (512, 1000), (544, 1000),
                                              (576, 1000), (608, 1000), (640, 1000),
                                              (672, 1000), (704, 1000), (736, 1000),
                                              (768, 1000), (800, 1000)],
                                'multiscale_mode': 'value',
                                'keep_ratio': True
                            },
                            {
                                'type': 'Rotate',
                                'prob': 0.3,
                                'level': 0,
                            },
                            {
                                'type': 'Shear',
                                'prob': 0.6,
                                'level': 8,
                                'direction': 'vertical',
                            }
                        ],
                        [
                            {
                                'type': 'Resize',
                                'img_scale': [(480, 1000), (512, 1000), (544, 1000),
                                              (576, 1000), (608, 1000), (640, 1000),
                                              (672, 1000), (704, 1000), (736, 1000),
                                              (768, 1000), (800, 1000)],
                                'multiscale_mode': 'value',
                                'keep_ratio': True,
                            },
                            {
                                'type': 'ColorTransform',
                                'prob': 0.5,
                                'level': 6,
                            },
                            {
                                'type': 'CutOut',
                                'n_holes': 5,
                                'cutout_shape': [(50, 50), (75, 80), (100, 100)],
                            },
                        ],
                        [
                            {
                                'type': 'Resize',
                                'img_scale': [(480, 1000), (512, 1000), (544, 1000),
                                              (576, 1000), (608, 1000), (640, 1000),
                                              (672, 1000), (704, 1000), (736, 1000),
                                              (768, 1000), (800, 1000)],
                                'multiscale_mode': 'value',
                                'keep_ratio': True
                            },
                            {
                                'type': 'ContrastTransform',
                                'prob': 0.4,
                                'level': 6,
                            },
                            {
                                'type': 'Shear',
                                'prob': 0.8,
                                'level': 8,
                                'direction': 'horizontal',
                            },
                            {
                                'type': 'BrightnessTransform',
                                'prob': 0.5,
                                'level': 10,
                            }
                        ],
                        [
                            {
                                'type': 'Resize',
                                'img_scale': [(480, 1000), (512, 1000), (544, 1000),
                                              (576, 1000), (608, 1000), (640, 1000),
                                              (672, 1000), (704, 1000), (736, 1000),
                                              (768, 1000), (800, 1000)],
                                'multiscale_mode': 'value',
                                'keep_ratio': True
                            },
                            {
                                'type': 'BrightnessTransform',
                                'prob': 1.0,
                                'level': 2,
                            },
                            {
                                'type': 'Translate',
                                'prob': 1.0,
                                'level': 6,
                                'direction': 'vertical',
                            }
                        ],
                        [
                            {
                                'type': 'Resize',
                                'img_scale': [(480, 1000), (512, 1000), (544, 1000),
                                              (576, 1000), (608, 1000), (640, 1000),
                                              (672, 1000), (704, 1000), (736, 1000),
                                              (768, 1000), (800, 1000)],
                                'multiscale_mode': 'value',
                                'keep_ratio': True
                            },
                            {
                                'type': 'Rotate',
                                'prob': 1.0,
                                'level': 10,
                            },
                            {
                                'type': 'CutOut',
                                'n_holes': 6,
                                'cutout_shape': [(10, 50), (30, 35), (40, 45)],
                            },
                        ],
                        [
                            {
                                'type': 'Resize',
                                'img_scale': [(480, 1000), (512, 1000), (544, 1000),
                                              (576, 1000), (608, 1000), (640, 1000),
                                              (672, 1000), (704, 1000), (736, 1000),
                                              (768, 1000), (800, 1000)],
                                'multiscale_mode': 'value',
                                'keep_ratio': True
                            },
                            {
                                'type': 'Translate',
                                'prob': 0.2,
                                'level': 2,
                                'direction': 'vertical',
                            },
                            {
                                'type': 'Shear',
                                'prob': 0.8,
                                'level': 8,
                                'direction': 'vertical',
                            },
                            {
                                'type': 'Rotate',
                                'prob': 0.8,
                                'level': 8,
                            }
                        ]
                    ]),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    ),
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
evaluation = dict(interval=1, metric=['mAP'])
