_base_ = [
    '../../mmaction2/configs/_base_/models/tsn_r50.py',
    '../../mmaction2/configs/_base_/schedules/sgd_100e.py',
    '../../mmaction2/configs/_base_/default_runtime.py'
]

model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=2,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(
            type='AvgConsensus',
            dim=1),
        dropout_ratio=0.4,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    train_cfg=None,
    test_cfg=None)

work_dir='work_dirs/tricks'

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'datasets/Tricks'
ann_file_train = 'tricks_train_video.txt'
ann_file_val = 'tricks_val_video.txt'
ann_file_test = 'tricks_test_video.txt'

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(
        type='DecordInit',
        **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=3),
    dict(
        type='DecordDecode'),
    dict(
        type='Resize',
        scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(
        type='Resize',
        scale=(224, 224),
        keep_ratio=False),
    dict(
        type='Flip',
        flip_ratio=0.5),
    dict(
        type='FormatShape',
        input_format='NCHW'),
    dict(
        type='PackActionInputs')
]

val_pipeline = [
    dict(
        type='DecordInit',
        **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=3,
        test_mode=True),
    dict(
        type='DecordDecode'),
    dict(
        type='Resize',
        scale=(-1, 256)),
    dict(
        type='CenterCrop',
        crop_size=224),
    dict(
        type='FormatShape',
        input_format='NCHW'),
    dict(
        type='PackActionInputs')
]

test_pipeline = [
    dict(
        type='DecordInit',
        **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    dict(
        type='DecordDecode'),
    dict(
        type='Resize',
        scale=(-1, 256)),
    dict(
        type='TenCrop',
        crop_size=224),
    dict(
        type='FormatShape',
        input_format='NCHW'),
    dict(
        type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_root=data_root,
        data_prefix=dict(video='train')))

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_root=data_root,
        data_prefix=dict(video='val'),
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        data_root=data_root,
        data_prefix=dict(video='test'),
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=5000,
    val_interval=20)

val_cfg = dict(
    type='ValLoop')

test_cfg = dict(
    type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=5000,
        by_epoch=True,
        milestones=[i * 50 for i in range (100)],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001),
    clip_grad=dict(
        max_norm=40,
        norm_type=2))

default_scope = 'mmaction'

default_hooks = dict(
    runtime_info=dict(
        type='RuntimeInfoHook'),
    timer=dict(
        type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=50,
        ignore_last=False),
    param_scheduler=dict(
        type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        max_keep_ckpts=1),
    sampler_seed=dict(
        type='DistSamplerSeedHook'),
    sync_buffers=dict(
        type='SyncBuffersHook'),
    visualization=dict(
        type="VisualizationHook"))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(
        mp_start_method='fork',
        opencv_num_threads=0),
    dist_cfg=dict(
        backend='nccl'))

log_processor = dict(
    type='LogProcessor',
    window_size=20,
    by_epoch=True)

vis_backends = [
    dict(
        type='LocalVisBackend')]

visualizer = dict(
    type='ActionVisualizer',
    vis_backends=vis_backends)

log_level = 'INFO'
