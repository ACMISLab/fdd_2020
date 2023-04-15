# model settings
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
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
                        [{
                            'type': 'Resize',
                            'img_scale': [(544, 1000), (640, 1000), (736, 1000)],
                            'multiscale_mode': 'value',
                            'keep_ratio': True
                        }, {
                            'type': 'MinIoURandomCrop',
                            'min_ious': (0.1, 0.3, 0.5, 0.7),
                        }, {
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
                        [{
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
                        [{
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
                        [{
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
                        [{
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
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=1)
log_config = dict(interval=200, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
