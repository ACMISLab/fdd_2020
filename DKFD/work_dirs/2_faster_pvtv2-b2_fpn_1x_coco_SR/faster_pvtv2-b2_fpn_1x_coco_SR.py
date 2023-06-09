model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='PyramidVisionTransformerV2',
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
        init_cfg=dict(
            checkpoint=
            'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth'
        )),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
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
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
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
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(type='DKLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
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
            max_per_img=100)))
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
        p=0.1)
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='VOCDataset',
            ann_file=[
                'data/VOCdevkit/VOC2007/ImageSets/Main/trainval_SR_USM.txt'
            ],
            img_prefix=['data/VOCdevkit/VOC2007/'],
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='AutoAugment',
                    policies=[[{
                        'type':
                        'Resize',
                        'img_scale': [(480, 1000), (512, 1000), (544, 1000),
                                      (576, 1000), (608, 1000), (640, 1000),
                                      (672, 1000), (704, 1000), (736, 1000),
                                      (768, 1000), (800, 1000)],
                        'multiscale_mode':
                        'value',
                        'keep_ratio':
                        True
                    }],
                              [{
                                  'type':
                                  'Resize',
                                  'img_scale': [(500, 1000), (600, 1000),
                                                (700, 1000)],
                                  'multiscale_mode':
                                  'value',
                                  'keep_ratio':
                                  True
                              }, {
                                  'type': 'RandomCrop',
                                  'crop_type': 'absolute_range',
                                  'crop_size': (400, 600),
                                  'allow_negative_crop': False
                              }, {
                                  'type':
                                  'Resize',
                                  'img_scale': [(480, 1000), (512, 1000),
                                                (544, 1000), (576, 1000),
                                                (608, 1000), (640, 1000),
                                                (672, 1000), (704, 1000),
                                                (736, 1000), (768, 1000),
                                                (800, 1000)],
                                  'multiscale_mode':
                                  'value',
                                  'override':
                                  True,
                                  'keep_ratio':
                                  True
                              }],
                              [{
                                  'type':
                                  'Resize',
                                  'img_scale': [(544, 1000), (640, 1000),
                                                (736, 1000)],
                                  'multiscale_mode':
                                  'value',
                                  'keep_ratio':
                                  True
                              }, {
                                  'type': 'MinIoURandomCrop',
                                  'min_ious': (0.1, 0.3, 0.5, 0.7)
                              }, {
                                  'type':
                                  'Resize',
                                  'img_scale': [(480, 1000), (512, 1000),
                                                (544, 1000), (576, 1000),
                                                (608, 1000), (640, 1000),
                                                (672, 1000), (704, 1000),
                                                (736, 1000), (768, 1000),
                                                (800, 1000)],
                                  'multiscale_mode':
                                  'value',
                                  'override':
                                  True,
                                  'keep_ratio':
                                  True
                              }],
                              [{
                                  'type':
                                  'Resize',
                                  'img_scale': [(480, 1000), (512, 1000),
                                                (544, 1000), (576, 1000),
                                                (608, 1000), (640, 1000),
                                                (672, 1000), (704, 1000),
                                                (736, 1000), (768, 1000),
                                                (800, 1000)],
                                  'multiscale_mode':
                                  'value',
                                  'keep_ratio':
                                  True
                              }, {
                                  'type': 'PhotoMetricDistortion'
                              }],
                              [{
                                  'type':
                                  'Resize',
                                  'img_scale': [(480, 1000), (512, 1000),
                                                (544, 1000), (576, 1000),
                                                (608, 1000), (640, 1000),
                                                (672, 1000), (704, 1000),
                                                (736, 1000), (768, 1000),
                                                (800, 1000)],
                                  'multiscale_mode':
                                  'value',
                                  'keep_ratio':
                                  True
                              }, {
                                  'type':
                                  'Albu',
                                  'transforms': [{
                                      'type': 'ShiftScaleRotate',
                                      'shift_limit': 0.0625,
                                      'scale_limit': 0.0,
                                      'rotate_limit': 0,
                                      'interpolation': 1,
                                      'p': 0.5
                                  }, {
                                      'type': 'RandomBrightnessContrast',
                                      'brightness_limit': [0.1, 0.3],
                                      'contrast_limit': [0.1, 0.3],
                                      'p': 0.2
                                  }, {
                                      'type':
                                      'OneOf',
                                      'transforms': [{
                                          'type': 'RGBShift',
                                          'r_shift_limit': 10,
                                          'g_shift_limit': 10,
                                          'b_shift_limit': 10,
                                          'p': 1.0
                                      }, {
                                          'type': 'HueSaturationValue',
                                          'hue_shift_limit': 20,
                                          'sat_shift_limit': 30,
                                          'val_shift_limit': 20,
                                          'p': 1.0
                                      }],
                                      'p':
                                      0.1
                                  }, {
                                      'type': 'ChannelShuffle',
                                      'p': 0.1
                                  }, {
                                      'type':
                                      'OneOf',
                                      'transforms': [{
                                          'type': 'Blur',
                                          'blur_limit': 3,
                                          'p': 1.0
                                      }, {
                                          'type': 'MedianBlur',
                                          'blur_limit': 3,
                                          'p': 1.0
                                      }],
                                      'p':
                                      0.1
                                  }],
                                  'bbox_params': {
                                      'type': 'BboxParams',
                                      'format': 'pascal_voc',
                                      'label_fields': ['gt_labels'],
                                      'min_visibility': 0.0,
                                      'filter_lost_elements': True
                                  },
                                  'keymap': {
                                      'img': 'image',
                                      'gt_bboxes': 'bboxes'
                                  },
                                  'update_pad_shape':
                                  False,
                                  'skip_img_without_anno':
                                  True
                              }],
                              [{
                                  'type':
                                  'Resize',
                                  'img_scale': [(480, 1000), (512, 1000),
                                                (544, 1000), (576, 1000),
                                                (608, 1000), (640, 1000),
                                                (672, 1000), (704, 1000),
                                                (736, 1000), (768, 1000),
                                                (800, 1000)],
                                  'multiscale_mode':
                                  'value',
                                  'keep_ratio':
                                  True
                              }, {
                                  'type': 'RandomShift'
                              }],
                              [{
                                  'type':
                                  'Resize',
                                  'img_scale': [(480, 1000), (512, 1000),
                                                (544, 1000), (576, 1000),
                                                (608, 1000), (640, 1000),
                                                (672, 1000), (704, 1000),
                                                (736, 1000), (768, 1000),
                                                (800, 1000)],
                                  'multiscale_mode':
                                  'value',
                                  'keep_ratio':
                                  True
                              }, {
                                  'type': 'RandomAffine'
                              }],
                              [{
                                  'type':
                                  'Resize',
                                  'img_scale': [(480, 1000), (512, 1000),
                                                (544, 1000), (576, 1000),
                                                (608, 1000), (640, 1000),
                                                (672, 1000), (704, 1000),
                                                (736, 1000), (768, 1000),
                                                (800, 1000)],
                                  'multiscale_mode':
                                  'value',
                                  'keep_ratio':
                                  True
                              }, {
                                  'type': 'CutOut',
                                  'n_holes': 3,
                                  'cutout_shape': [(10, 20), (15, 30),
                                                   (20, 40)]
                              }],
                              [{
                                  'type':
                                  'Resize',
                                  'img_scale': [(480, 1000), (512, 1000),
                                                (544, 1000), (576, 1000),
                                                (608, 1000), (640, 1000),
                                                (672, 1000), (704, 1000),
                                                (736, 1000), (768, 1000),
                                                (800, 1000)],
                                  'multiscale_mode':
                                  'value',
                                  'keep_ratio':
                                  True
                              }, {
                                  'type': 'ColorTransform',
                                  'prob': 0.8,
                                  'level': 4
                              }, {
                                  'type': 'Translate',
                                  'prob': 1.0,
                                  'level': 6,
                                  'direction': 'vertical'
                              }, {
                                  'type': 'Rotate',
                                  'prob': 0.6,
                                  'level': 6
                              }],
                              [{
                                  'type':
                                  'Resize',
                                  'img_scale': [(480, 1000), (512, 1000),
                                                (544, 1000), (576, 1000),
                                                (608, 1000), (640, 1000),
                                                (672, 1000), (704, 1000),
                                                (736, 1000), (768, 1000),
                                                (800, 1000)],
                                  'multiscale_mode':
                                  'value',
                                  'keep_ratio':
                                  True
                              }, {
                                  'type': 'Rotate',
                                  'prob': 0.3,
                                  'level': 0
                              }, {
                                  'type': 'Shear',
                                  'prob': 0.6,
                                  'level': 8,
                                  'direction': 'vertical'
                              }],
                              [{
                                  'type':
                                  'Resize',
                                  'img_scale': [(480, 1000), (512, 1000),
                                                (544, 1000), (576, 1000),
                                                (608, 1000), (640, 1000),
                                                (672, 1000), (704, 1000),
                                                (736, 1000), (768, 1000),
                                                (800, 1000)],
                                  'multiscale_mode':
                                  'value',
                                  'keep_ratio':
                                  True
                              }, {
                                  'type': 'ColorTransform',
                                  'prob': 0.5,
                                  'level': 6
                              }, {
                                  'type': 'CutOut',
                                  'n_holes': 5,
                                  'cutout_shape': [(50, 50), (75, 80),
                                                   (100, 100)]
                              }],
                              [{
                                  'type':
                                  'Resize',
                                  'img_scale': [(480, 1000), (512, 1000),
                                                (544, 1000), (576, 1000),
                                                (608, 1000), (640, 1000),
                                                (672, 1000), (704, 1000),
                                                (736, 1000), (768, 1000),
                                                (800, 1000)],
                                  'multiscale_mode':
                                  'value',
                                  'keep_ratio':
                                  True
                              }, {
                                  'type': 'ContrastTransform',
                                  'prob': 0.4,
                                  'level': 6
                              }, {
                                  'type': 'Shear',
                                  'prob': 0.8,
                                  'level': 8,
                                  'direction': 'horizontal'
                              }, {
                                  'type': 'BrightnessTransform',
                                  'prob': 0.5,
                                  'level': 10
                              }],
                              [{
                                  'type':
                                  'Resize',
                                  'img_scale': [(480, 1000), (512, 1000),
                                                (544, 1000), (576, 1000),
                                                (608, 1000), (640, 1000),
                                                (672, 1000), (704, 1000),
                                                (736, 1000), (768, 1000),
                                                (800, 1000)],
                                  'multiscale_mode':
                                  'value',
                                  'keep_ratio':
                                  True
                              }, {
                                  'type': 'BrightnessTransform',
                                  'prob': 1.0,
                                  'level': 2
                              }, {
                                  'type': 'Translate',
                                  'prob': 1.0,
                                  'level': 6,
                                  'direction': 'vertical'
                              }],
                              [{
                                  'type':
                                  'Resize',
                                  'img_scale': [(480, 1000), (512, 1000),
                                                (544, 1000), (576, 1000),
                                                (608, 1000), (640, 1000),
                                                (672, 1000), (704, 1000),
                                                (736, 1000), (768, 1000),
                                                (800, 1000)],
                                  'multiscale_mode':
                                  'value',
                                  'keep_ratio':
                                  True
                              }, {
                                  'type': 'Rotate',
                                  'prob': 1.0,
                                  'level': 10
                              }, {
                                  'type': 'CutOut',
                                  'n_holes': 6,
                                  'cutout_shape': [(10, 50), (30, 35),
                                                   (40, 45)]
                              }],
                              [{
                                  'type':
                                  'Resize',
                                  'img_scale': [(480, 1000), (512, 1000),
                                                (544, 1000), (576, 1000),
                                                (608, 1000), (640, 1000),
                                                (672, 1000), (704, 1000),
                                                (736, 1000), (768, 1000),
                                                (800, 1000)],
                                  'multiscale_mode':
                                  'value',
                                  'keep_ratio':
                                  True
                              }, {
                                  'type': 'Translate',
                                  'prob': 0.2,
                                  'level': 2,
                                  'direction': 'vertical'
                              }, {
                                  'type': 'Shear',
                                  'prob': 0.8,
                                  'level': 8,
                                  'direction': 'vertical'
                              }, {
                                  'type': 'Rotate',
                                  'prob': 0.8,
                                  'level': 8
                              }]]),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
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
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
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
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
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
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=200, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='SetEpochInfoHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
work_dir = './work_dirs/faster_pvtv2-b2_fpn_1x_coco_SR'
auto_resume = False
gpu_ids = [0]
