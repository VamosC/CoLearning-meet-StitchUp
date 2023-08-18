# model settings
gpus=1
freq_file = 'appendix/VOCdevkit/longtail2012/class_freq_0.5noise.pkl'
model = dict(
    type='StitchUpClassifier',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='PFC',
        in_channels=2048,
        out_channels=256,
        dropout=0),
    neck1=dict(
        type='PFC',
        in_channels=2048,
        out_channels=256,
        dropout=0.5),
    head1=dict(
        type='ClsHead',
        in_channels=256,
        num_classes=20,
        method='fc',
        loss_cls=dict(
            type='ResampleLoss', use_sigmoid=True,
            reweight_func='rebalance',
            focal=dict(focal=True, balance_param=2.0, gamma=2),
            logit_reg=dict(init_bias=0.05, neg_scale=5),
            map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
            loss_weight=1.0,
            freq_file=freq_file)),
    head=dict(
        type='ClsHead',
        in_channels=256,
        num_classes=20,
        method='fc',
        loss_cls=dict(
            type='BCELoss', use_sigmoid=True,
            reweight_func='rebalance',
            focal=dict(focal=False, balance_param=2.0, gamma=2),
            logit_reg=dict(init_bias=0.05, neg_scale=5),
            map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
            loss_weight=1.0,
            freq_file=freq_file)))
# model training and testing settings
online_data_root = 'appendix/VOCdevkit/'
train_cfg = dict(class_split=online_data_root + 'longtail2012/training_split.pkl',
                 warmup_epoch=0,
                 random_head_alpha=1.0,
                 random_medium_alpha=1.0,
                 random_tail_alpha=1.0,
                 random_head_beta=0.0,
                 random_medium_beta=0.0,
                 random_tail_beta=0.0,
                 balance_head_alpha=1.0,
                 balance_medium_alpha=1.0,
                 balance_tail_alpha=1.0,
                 balance_head_beta=0.0,
                 balance_medium_beta=0.0,
                 balance_tail_beta=0.0)
test_cfg = dict(alpha=0.1,
                dual=True)

# dataset settings
noise_dataset_type = 'VOCNoiseDataset'
dataset_type = 'VOCDataset'
data_root = 'data/voc/VOCdevkit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
extra_aug = dict(
    photo_metric_distortion=dict(
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    random_crop=dict(
        min_crop_size=0.8
    )
)
data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=2,
    sampler='None',
    sampler1='ClassAware',
    train=dict(
            type=noise_dataset_type,
            noise_label_file=freq_file,
            ann_file=online_data_root + 'longtail2012/img_id.txt',
            img_prefix=data_root + 'VOC2012/',
            img_scale=(224, 224),
            img_norm_cfg=img_norm_cfg,
            extra_aug=extra_aug,
            size_divisor=32,
            resize_keep_ratio=False,
            flip_ratio=0.5
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        class_split=online_data_root + 'longtail2012/class_split.pkl',
        img_scale=(224, 224),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        resize_keep_ratio=False,
        flip_ratio=0))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    step=[5, 7])
checkpoint_config = dict(interval=8)
# yapf:disable
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
evaluation = dict(interval=5)
# runtime settings
total_epochs = 8
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/voc_resnet50'
load_from = None
resume_from = None
workflow = [('train', 1)]
